"""Verify observations, teacher projections, fixed geometry and actual-output labels."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from navigation_common import Curve,save
from direct_frontend import observation,net_class,decode
from build_direct_teachers import fit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1);d=np.load(a.dataset/'dataset.npz');rows=json.loads((a.dataset/'records.json').read_text());protocol=json.loads((a.dataset/'protocol.json').read_text())
    parent=a.dataset.parent/Path(protocol['parent_dataset']).name
    assert hashlib.sha256((parent/'dataset.npz').read_bytes()).hexdigest()==protocol['parent_sha256']
    models={};labels={};source_cache={};cloud_count=0;teacher_count=0
    manifest=json.loads((a.labels/'protocol.json').read_text())['model_manifest']
    for seed in [11,22,33]:
        p=a.models/f'direct_s{seed}.pt';final=torch.load(p,map_location='cpu',weights_only=True);geometry=torch.load(a.models/f'geometry_s{seed}.pt',map_location='cpu',weights_only=True)
        assert hashlib.sha256((a.models/f'geometry_s{seed}.pt').read_bytes()).hexdigest()==manifest[str(seed)]['sha256']==final['geometry_sha256']
        assert all(torch.equal(value,final['state_dict'][name]) for name,value in geometry['state_dict'].items() if not name.startswith('scorer.'))
        model=net_class()().eval();model.load_state_dict(final['state_dict']);models[seed]=model;labels[seed]=np.load(a.labels/f'labels_s{seed}.npz')
    for i,row in enumerate(rows):
        case=row['case'];file=parent/'source_logs'/(case+'.jsonl')
        if case not in source_cache:
            assert hashlib.sha256(file.read_bytes()).hexdigest()==row['source_sha256'];source_cache[case]=[json.loads(l) for l in file.read_text().splitlines()]
        entry=source_cache[case][row['source_entry']];q=row['query']
        assert all(entry['request'][k]==q[k] for k in ['t','start_state','goal'])
        cloud=dict(row['cloud']);cloud['points']=np.fromfile(a.dataset/cloud['binary_xyz_f32'],dtype='<f4').reshape(-1,3)
        assert cloud['stamp']<=q['t'];im,st=observation(cloud,q,row['backend']);assert np.array_equal(im,d['image'][i]) and np.array_equal(st,d['state'][i]);cloud_count+=1
        for j in np.flatnonzero(d['mask'][i]):
            original=entry['response']['plan'] if row['near_goal'] else entry['response']['branches'][j]['plan']
            target,_=fit(original,q);assert target is not None and np.max(np.abs(target-d['target'][i,j]))<1e-6;teacher_count+=1
        for seed,model in models.items():
            with torch.inference_mode():raw,_=model(torch.from_numpy(im[None]),torch.from_numpy(st[None]))
            assert np.max(np.abs(raw.numpy()[0]-labels[seed]['parameters'][i]))<1e-6
    replays=json.loads((a.dataset/'replays.json').read_text());evaluated=0;calls=0
    for case,replay in replays.items():
        logs=[json.loads(l) for l in (a.labels/'workers'/case/'requests.jsonl').read_text().splitlines()];cursor=0
        for event in replay['sequence']:
            if 'sample' not in event:
                q,r=logs[cursor]['request'],logs[cursor]['response'];cursor+=1
                assert q['op']=='update' and q['t']==event['t'] and q['cloud']['stamp']==event['cloud']['stamp']
                expected=(a.dataset/event['cloud']['binary_xyz_f32']).resolve();actual=Path(q['cloud']['binary_xyz_f32'])
                if actual.exists():assert hashlib.sha256(actual.read_bytes()).digest()==hashlib.sha256(expected.read_bytes()).digest()
                assert r['optimizer_calls']==0;continue
            i=event['sample'];row=rows[i]
            for seed in [11,22,33]:
                q,r=logs[cursor]['request'],logs[cursor]['response'];cursor+=1
                assert q['op']=='evaluate_direct' and r['optimizer_calls']==0;calls+=1
                assert all(q[k]==row['query'][k] for k in ['t','start_state','goal'])
                rebuilt,_=decode(q,labels[seed]['parameters'][i],np.zeros((15,2),np.float32));by_id={v['id']:v['plan'] for v in rebuilt}
                for v in q['direct_plans']:
                    for key in ['start','dt','duration','controls']:assert np.allclose(by_id[v['id']][key],v['plan'][key],atol=1e-12,rtol=0)
                for b in r['branches']:
                    j=b['id'];assert labels[seed]['feasible'][i,j]==float(b['ok'])
                    value=-2.
                    if b['ok']:
                        c=Curve(b['plan']);goal=np.asarray(q['goal']);initial=np.linalg.norm(goal-q['start_state']['p']);points=c.s(np.linspace(0,c.duration,61))
                        value=(initial-np.linalg.norm(goal-c.s(min(.6,c.duration))))/.6+.12*(initial-np.linalg.norm(goal-c.s(c.duration)))-.03*c.duration-.03*np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
                    assert abs(labels[seed]['quality'][i,j]-value)<1e-6;evaluated+=1
        assert cursor==len(logs)
    result=dict(verified=True,states=len(rows),observations_reconstructed=cloud_count,teacher_projections_verified=teacher_count,actual_curves_evaluated=evaluated,
        requests_with_zero_optimizer_calls=calls,geometry_unchanged_during_score_training=True,labels_recomputed=True,
        limits='Current-map sampled curve and stopping backup feasibility; not a long-horizon risk calibration.')
    save(a.out,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
