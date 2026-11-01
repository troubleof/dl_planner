"""Distil same-state EGO branches into a scored multi-trajectory proposal net."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from navigation_common import Curve,save
from adapted_frontends import inputs
from trajectory_frontend import model_class, geometry, DELTA_SCALE, FRACTIONS, FORMAT


def build(dataset, out):
    out.mkdir(parents=True,exist_ok=False)
    records = json.loads((dataset/'records.json').read_text())
    original = np.load(dataset/'dataset.npz')
    caches, pointers = {}, {}
    targets, masks, clipped, provenance = [], [], [], []
    source_hashes = {}
    for number, row in enumerate(records):
        case = row['case']
        if case not in caches:
            path = dataset/'branches'/case/'requests.jsonl'
            source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            latest = None
            evaluations = []
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                q = rec['request']
                if 'cloud' in q:
                    latest = dict(q['cloud'])
                if q['op']=='evaluate_routes':
                    evaluations.append((q,rec['response']['branches'],latest))
            caches[case] = evaluations
            pointers[case] = 0
        q, branches, cloud = caches[case][pointers[case]]
        pointers[case] += 1
        assert q['t']==row['t'] and cloud['stamp']==row['stamp']
        assert [b['ok'] for b in branches]==row['valid']
        cloud = dict(cloud)
        cloud['points'] = np.fromfile(cloud['binary_xyz_f32'],dtype='<f4').reshape(-1,3)
        im, st = inputs(cloud,q)
        assert np.array_equal(im,original['image'][number])
        # Historical binary cloud records retain capture_state, so verify state too.
        assert np.array_equal(st,original['state'][number])
        anchors, frames = geometry(q)
        target = np.zeros((15,3,3),np.float32)
        mask = np.zeros(15,np.float32)
        for action,b in enumerate(branches[:15]):
            if not b['ok']:
                continue
            curve = Curve(b['plan'])
            times = np.linspace(0,curve.duration,129)
            positions = np.asarray([curve.state(curve.start+t)['p'] for t in times])
            arc = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(positions,axis=0),axis=1))]
            if arc[-1]<.05:
                continue
            points = np.stack([np.interp(FRACTIONS*arc[-1],arc,positions[:,k]) for k in range(3)],axis=1)
            residual = ((points-anchors[action])@frames[action])/DELTA_SCALE
            clipped.extend((np.abs(residual)>1).ravel().tolist())
            target[action] = np.clip(residual,-1,1)
            mask[action] = 1
        targets.append(target)
        masks.append(mask)
        provenance.append(dict(row,request_t=q['t'],candidate_teachers=int(mask.sum())))
    assert all(pointers[k]==len(v) for k,v in caches.items())
    np.savez_compressed(out/'dataset.npz',image=original['image'],state=original['state'],
        quality=original['quality'],feasible=original['feasible'],offset=np.stack(targets),mask=np.stack(masks))
    save(out/'records.json',provenance)
    save(out/'protocol.json',dict(format=FORMAT,source=str(dataset),source_hashes=source_hashes,
        samples=len(records),train=sum(r['split']=='train' for r in records),
        validation=sum(r['split']=='validation' for r in records),
        target='3 equal-arc samples from each valid EGO candidate, local bounded offsets; invalid geometry masked',
        quality='Existing 0.6s progress + terminal progress - route length surrogate; not long-horizon value',
        clipped_coordinate_fraction=float(np.mean(clipped)),truth_in_observation=False,
        dataset_sha256=hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()))
    print(json.dumps(json.loads((out/'protocol.json').read_text())|{'source_hashes':'see protocol.json'}),flush=True)


def train(dataset,out):
    import torch
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    data=np.load(dataset/'dataset.npz')
    rows=json.loads((dataset/'records.json').read_text())
    ti=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    assert len(ti) and len(vi) and len(ti)+len(vi)==len(rows)
    assert not ({rows[i]['seed'] for i in ti}&{rows[i]['seed'] for i in vi})
    x,s,q,v,target,mask=[torch.from_numpy(data[k].astype(np.float32)) for k in ['image','state','quality','feasible','offset','mask']]
    def losses(model,ix):
        pred,logits,delta=model(x[ix],s[ix])
        score_loss=torch.nn.functional.smooth_l1_loss(pred,q[ix])
        feasibility_loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,v[ix])
        rank_loss=(-torch.softmax(q[ix]/.2,dim=1)*torch.log_softmax(pred/.2,dim=1)).sum(dim=1).mean()
        geometry_loss=((delta-target[ix]).square().mean(dim=(2,3))*mask[ix]).sum()/mask[ix].sum().clamp_min(1.)
        total=score_loss+.3*feasibility_loss+.15*rank_loss+2.*geometry_loss
        return total,geometry_loss,pred,logits
    histories=[]
    for seed in [11,22,33]:
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        model=model_class()()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float('inf')
        history=[]
        begin=time.perf_counter()
        for epoch in range(1,101):
            model.train()
            order=rng.permutation(ti)
            for offset in range(0,len(order),32):
                loss,*_=losses(model,order[offset:offset+32])
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                loss,gl,pred,logits=losses(model,vi)
                selected=(pred-2*(1-torch.sigmoid(logits))).argmax(dim=1)
                regret=(q[vi].max(dim=1).values-q[vi,selected]).mean().item()
                metric=regret+2*gl.item()+.1*torch.nn.functional.smooth_l1_loss(pred,q[vi]).item()
            entry=dict(epoch=epoch,validation_regret=regret,validation_geometry_mse=gl.item(),selection_metric=metric)
            history.append(entry)
            if metric<best:
                best=metric;chosen=entry
                torch.save(dict(format=FORMAT,state_dict=model.state_dict(),seed=seed,epoch=epoch,
                    dataset_sha256=hashlib.sha256((dataset/'dataset.npz').read_bytes()).hexdigest()),out/f'trajectory_s{seed}.pt')
        histories.append(dict(seed=seed,parameters=sum(p.numel() for p in model.parameters()),
            training_samples=len(ti),validation_samples=len(vi),selected=chosen,seconds=time.perf_counter()-begin,history=history))
        save(out/'training.json',histories)
        print(json.dumps({k:v for k,v in histories[-1].items() if k!='history'}),flush=True)
    save(out/'data_manifest.json',json.loads((dataset/'protocol.json').read_text()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['build','train'])
    ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();(build if a.stage=='build' else train)(a.dataset,a.out)
