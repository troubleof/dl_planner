"""Same-observation branch labels, grouped split, three independent training seeds."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from navigation_common import ROOT,Worker,Curve,save
from adapted_frontends import inputs,primitives,model_class

def quality(branch,q):
    if not branch['ok']:return -2.
    curve=Curve(branch['plan']);goal=np.array(q['goal']);p=np.array(q['start_state']['p']);d=np.linalg.norm(p-goal)
    progress=d-np.linalg.norm(np.array(curve.state(curve.start+.6)['p'])-goal)
    endpoint=np.array(curve.state(curve.start+curve.duration)['p']);route=np.array(branch['route'])
    length=np.linalg.norm(np.diff(route,axis=0),axis=1).sum()
    # Local surrogate, NOT an optimal-action or success label. Matched to the
    # actual 0.6 s nominal segment preceding the precommitted backup branch.
    return float(progress/.6+.25*(d-np.linalg.norm(endpoint-goal))/curve.duration-.03*length)

def collect(run,out):
    out.mkdir(parents=True,exist_ok=False);images=[];states=[];targets=[];feasible=[];records=[]
    for row in json.loads((run/'summary.json').read_text()):
        folder=run/row['folder'];log=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()];latest=None;counter=0
        with Worker(row['backend'],out/'branches'/row['folder'],route_policy='repair') as w:
            for step,item in enumerate(log):
                q=item['request']
                if 'cloud' in q:
                    c=dict(q['cloud']);p=folder/'worker/clouds'/Path(c['binary_xyz_f32']).name
                    c['binary_xyz_f32']=str(p.resolve());w.request(dict(op='update',t=q['t'],cloud=c))
                    latest=dict(c,points=np.fromfile(p,dtype='<f4').reshape(-1,3).tolist())
                if q['op']!='plan':continue
                counter+=1
                # Original repair plan calls maintain the original cache state.
                query={k:q[k] for k in ['t','start_state','goal','with_backup']}
                if counter%2==0 and latest is not None and np.linalg.norm(np.array(q['goal'])-q['start_state']['p'])>=3. and q['t']-latest['stamp']<=.5:
                    image,state=inputs(latest,q);response=w.request(dict(query,op='evaluate_routes',candidates=primitives(q)))
                    branches=response['branches'];assert len(branches)==16
                    values=np.array([quality(b,q) for b in branches]);valid=np.array([b['ok'] for b in branches])
                    images.append(image);states.append(state);targets.append(values);feasible.append(valid)
                    records.append(dict(case=row['folder'],split=row['split'],seed=row['seed'],backend=row['backend'],step=step,stamp=latest['stamp'],t=q['t'],
                        labels=values.tolist(),valid=valid.tolist(),selected=int(values.argmax())))
                w.request(dict(query,op='plan'))
        print(json.dumps(dict(case=row['folder'],samples=len(records))),flush=True)
    np.savez_compressed(out/'dataset.npz',image=np.stack(images),state=np.stack(states),quality=np.stack(targets).astype(np.float32),feasible=np.stack(feasible).astype(np.float32))
    save(out/'records.json',records);save(out/'protocol.json',dict(source_run=str(run),samples=len(records),label='EGO/backup feasibility plus short-term progress surrogate',uses_scene_truth=False,
        split_by_map=True,action15='repair',weights='no YOPO initialization; small scorer trained from scratch',dataset_sha256=hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()))

def train(dataset,out):
    import torch
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False);data=np.load(dataset/'dataset.npz');records=json.loads((dataset/'records.json').read_text())
    trainmask=np.array([r['split']=='train' for r in records]);valmask=~trainmask
    assert trainmask.any() and valmask.any()
    x=torch.from_numpy(data['image']);s=torch.from_numpy(data['state']);q=torch.from_numpy(data['quality']);valid=torch.from_numpy(data['feasible'])
    ti=np.flatnonzero(trainmask);vi=np.flatnonzero(valmask);histories=[]
    for seed in [11,22,33]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=model_class()();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        best=float('inf');history=[];begin=time.perf_counter()
        for epoch in range(1,81):
            model.train();order=rng.permutation(ti)
            for offset in range(0,len(order),32):
                ix=order[offset:offset+32];pred,logit=model(x[ix],s[ix]);
                loss=torch.nn.functional.smooth_l1_loss(pred,q[ix])+.3*torch.nn.functional.binary_cross_entropy_with_logits(logit,valid[ix])
                # Pairwise action preference derived from all evaluated actions.
                target=torch.softmax(q[ix]/.2,dim=1);loss+=.15*(-target*torch.log_softmax(pred/.2,dim=1)).sum(dim=1).mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            model.eval()
            with torch.inference_mode():
                pred,logit=model(x[vi],s[vi]);utility=pred-(1-torch.sigmoid(logit))*2
                chosen=utility.argmax(dim=1);regret=(q[vi].max(dim=1).values-q[vi,chosen]).mean().item()
                mse=torch.nn.functional.smooth_l1_loss(pred,q[vi]).item();metric=regret+.1*mse
                entry=dict(epoch=epoch,validation_regret=regret,validation_smooth_l1=mse,validation_choice_feasible=float(valid[vi,chosen].mean()),validation_repair_fraction=float((chosen==15).float().mean()))
            history.append(entry)
            if metric<best:
                best=metric;torch.save(dict(state_dict=model.state_dict(),seed=seed,epoch=epoch),out/f'range_scorer_s{seed}.pt');selected=dict(entry)
        histories.append(dict(seed=seed,parameters=sum(p.numel() for p in model.parameters()),training_samples=int(trainmask.sum()),validation_samples=int(valmask.sum()),selected=selected,seconds=time.perf_counter()-begin,history=history))
        save(out/'training.json',histories);print(json.dumps({k:v for k,v in histories[-1].items() if k!='history'}),flush=True)
    save(out/'data_manifest.json',json.loads((dataset/'protocol.json').read_text()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    c=sub.add_parser('collect');c.add_argument('--run',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    t=sub.add_parser('train');t.add_argument('--dataset',type=Path,required=True);t.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.command=='collect':collect(a.run,a.out)
    else:train(a.dataset,a.out)
