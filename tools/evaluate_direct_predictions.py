"""Reevaluate actual frozen geometry against replayed arrived-cloud maps; no EGO calls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from navigation_common import Worker,Curve,save
from direct_frontend import net_class,decode,PARAMS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((a.dataset/'records.json').read_text());d=np.load(a.dataset/'dataset.npz');replays=json.loads((a.dataset/'replays.json').read_text())
    models={};parameters={};quality={};feasible={};manifest={};details=[]
    for seed in [11,22,33]:
        path=a.models/f'geometry_s{seed}.pt';ckpt=torch.load(path,map_location='cpu',weights_only=True);model=net_class()().eval();model.load_state_dict(ckpt['state_dict']);models[seed]=model
        parameters[seed]=np.zeros((len(rows),15,PARAMS),np.float32);quality[seed]=np.full((len(rows),15),-2.,np.float32);feasible[seed]=np.zeros((len(rows),15),np.float32)
        manifest[str(seed)]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    begin=time.perf_counter();seen=set();reasons={}
    for case,replay in replays.items():
        with Worker(replay['backend'],a.out/'workers'/case) as worker:
            for event in replay['sequence']:
                if 'sample' not in event:
                    q=dict(event);q['cloud']=dict(q['cloud']);q['cloud']['binary_xyz_f32']=str((a.dataset/q['cloud']['binary_xyz_f32']).resolve());r=worker.request(q)
                    assert r['optimizer_calls']==0;continue
                i=event['sample'];seen.add(i);row=rows[i];q=row['query'];state=d['state'][i];image=d['image'][i]
                for seed,model in models.items():
                    with torch.inference_mode():raw,pred=model(torch.from_numpy(image[None]),torch.from_numpy(state[None]))
                    raw,pred=raw.numpy()[0],pred.numpy()[0];plans,_=decode(q,raw,pred);parameters[seed][i]=raw
                    r=worker.request(dict(q,op='evaluate_direct',direct_plans=plans,forbid_optimizer=True))
                    assert r['optimizer_calls']==0 and r['ok']
                    for branch in r['branches']:
                        j=branch['id'];why=branch['reason'];reasons[why]=reasons.get(why,0)+1
                        if not branch['ok']:continue
                        c=Curve(branch['plan']);goal=np.asarray(q['goal']);initial=np.linalg.norm(goal-q['start_state']['p'])
                        progress=(initial-np.linalg.norm(goal-c.s(min(.6,c.duration))))/.6
                        endpoint=initial-np.linalg.norm(goal-c.s(c.duration))
                        ps=c.s(np.linspace(0,c.duration,61));length=np.linalg.norm(np.diff(ps,axis=0),axis=1).sum()
                        value=progress+.12*endpoint-.03*c.duration-.03*length
                        quality[seed][i,j]=value;feasible[seed][i,j]=1
                    details.append(dict(state=i,seed=seed,reasons={b['id']:b['reason'] for b in r['branches']}))
        print(json.dumps(dict(case=case,states_completed=len(seen),seconds=time.perf_counter()-begin)),flush=True)
    assert seen==set(range(len(rows)))
    for seed in models:np.savez_compressed(a.out/f'labels_s{seed}.npz',parameters=parameters[seed],quality=quality[seed],feasible=feasible[seed])
    save(a.out/'details.json',details);save(a.out/'protocol.json',dict(states=len(rows),curves=len(rows)*45,model_manifest=manifest,reasons=reasons,
        geometry_dataset_sha256=hashlib.sha256((a.dataset/'dataset.npz').read_bytes()).hexdigest(),online_optimizer_calls=0,
        labels='actual predicted curve passes unchanged whole-curve and stopping backup checks; quality is .6s progress + .12 terminal progress - .03 duration - .03 arc length',
        limitation='finite sampled map and current stop feasibility, not multi-step recovery or future-map safety'))

if __name__=='__main__':main()
