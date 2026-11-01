"""Actual camera-policy states; offline timing branches on their received maps."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from navigation_common import Worker,Curve,save
from direct_frontend import decode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stride',type=int,default=5);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=[];replays={};arrays={k:[] for k in ['image','state','seed','features','target','mask']};counts=Counter();details=[];begin=time.perf_counter()
    for run in a.runs:
        for result in json.loads((run/'summary.json').read_text()):
            assert result['split'] in ['train','validation'],'test data forbidden'
            case=run/result['folder'];name=run.name+'__'+result['folder'];records=[json.loads(x) for x in (case/'worker/requests.jsonl').read_text().splitlines()];sequence=[];latest=None;number=0
            with Worker(result['backend'],a.out/'workers'/name) as w:
                for index,item in enumerate(records,1):
                    q=item['request']
                    if 'cloud' in q:
                        latest=q['cloud'];ev=dict(op='update',t=q['t'],cloud=latest);w.request(ev);sequence.append(ev)
                    if q['op']!='plan':continue
                    number+=1
                    if number>8 and number%a.stride:continue
                    evidence=case/'worker/neural'/f'{index:05d}.npz'
                    if not evidence.exists():counts['missing_input']+=1;continue
                    data=np.load(evidence);seed=data['seed'];query={k:q[k] for k in ['t','start_state','goal','local_progress_target','map_context','with_backup','execution_delay_s'] if k in q}
                    target=seed.copy();mask=np.zeros(15,np.float32);best=np.full(15,-np.inf);branches=[];diag=[]
                    for variant,shift in enumerate([-.18,0.,.18]):
                        raw=seed.copy();raw[:,0]=np.clip(raw[:,0]+shift,np.log(.6),np.log(5.95));plans,_=decode(query,raw,np.zeros((15,2)))
                        checked=w.request(dict(query,op='evaluate_direct',direct_plans=plans,forbid_optimizer=True));assert checked['optimizer_calls']==0
                        for b in checked['branches']:
                            j=b['id'];counts[b['reason']]+=1;diag.append(dict(candidate=j,timing_variant=variant,reason=b['reason']))
                            if not b['ok']:continue
                            c=Curve(b['plan']);goal=np.asarray(query.get('local_progress_target',query['goal']));d=np.linalg.norm(goal-query['start_state']['p']);progress=(d-np.linalg.norm(goal-c.s(min(.6,c.duration))))/.6
                            score=progress+.12*(d-np.linalg.norm(goal-c.s(c.duration)))-.03*c.duration
                            if score>best[j]:best[j]=score;mask[j]=1;target[j]=raw[j]
                    i=len(rows);sequence.append(dict(sample=i));rows.append(dict(index=i,case=name,backend=result['backend'],seed=result['seed'],split=result['split'],policy=result['policy'],outcome=result['outcome'],query=query,cloud=latest,request_index=index,source_case=str(case.resolve())))
                    for key,value in [('image',data['range_image']),('state',data['state']),('seed',seed),('features',data['features']),('target',target),('mask',mask)]:arrays[key].append(value)
                    details.append(dict(state=i,branches=diag))
            replays[name]=dict(backend=result['backend'],sequence=sequence)
            print(json.dumps(dict(case=name,states=len(rows),seconds=time.perf_counter()-begin)),flush=True)
    np.savez_compressed(a.out/'dataset.npz',**{k:np.asarray(v,np.float32) for k,v in arrays.items()});save(a.out/'records.json',rows);save(a.out/'replays.json',replays);save(a.out/'teacher_checks.json',details)
    save(a.out/'protocol.json',dict(states=len(rows),train=sum(r['split']=='train' for r in rows),validation=sum(r['split']=='validation' for r in rows),valid_teachers=int(np.sum(arrays['mask'])),checks=dict(counts),sources=[str(p.resolve()) for p in a.runs],dataset_sha256=hashlib.sha256((a.out/'dataset.npz').read_bytes()).hexdigest(),teacher='Three offline timing choices, unchanged exact PVA decoding, full curve and stop validation on only previously received camera maps. No geometry optimizer; endpoint and residual targets are the known-free seeds.',sampling='First eight plan requests per episode then every fifth, including failed states. Only states having valid camera/context input can produce tensors.',limitations='Known-map finite stopping probes are not future-map recoverability or reinforcement-learning returns.'))

if __name__=='__main__':main()
