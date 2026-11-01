"""Censored future-map labels, using only train/validation camera-policy logs.

Future maps are privileged labels. Inputs use ONLY the received prefix. These
first labels follow the logged camera path; candidate-specific branches are
added separately, rather than pretending an off-policy image is on-policy.
"""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from navigation_common import Worker,Curve,save
from direct_frontend import decode
from recovery_frontend import inputs,HORIZONS,K,candidate_bank

def load_cloud(c):
    c=dict(c);c['points']=np.fromfile(c['binary_xyz_f32'],dtype='<f4').reshape(-1,3)
    dep=np.load(c['depth_evidence_npz']);c['depth_grid_world']=dep['world'];c['depth_valid']=dep['valid']
    return c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stride',type=int,default=5);ap.add_argument('--historical-failures-as',choices=['train','validation']);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    arrays={k:[] for k in ['image','previous','state','features','seed','labels','mask']};rows=[];details=[];counts=Counter();begin=time.perf_counter()
    for run in a.runs:
        for result in json.loads((run/'summary.json').read_text()):
            if a.historical_failures_as:
                assert run.name in ['visible-test-v1','visible-test-v2'],'only explicitly named historical frozen failures may be reassigned'
                if result['outcome']!='no_validated_escape':continue
                result=dict(result,original_split=result['split'],split=a.historical_failures_as)
            assert result['split'] in ['train','validation'],'fresh test data forbidden'
            folder=run/result['folder'];records=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()];candidates=[i for i,x in enumerate(records) if x['request']['op']=='plan' and x['request'].get('context_ok') and len((x['request'].get('map_context') or {}).get('endpoints',[]))==15]
            selected=set(candidates[:4]+candidates[4::a.stride])
            if result['outcome']!='reached':selected.update(candidates[-8:])
            latest=previous=None;pending=[];sequence=[];name=run.name+'__'+result['folder']
            def finish(t,inclusive):
                for p in pending:
                    for h,delta in enumerate(HORIZONS,1):
                        deadline=p['t']+delta
                        if not p['done'][h] and (deadline<t-1e-9 or inclusive and deadline<=t+1e-9):
                            arrays['labels'][p['id']][:,h]=p['alive'];arrays['mask'][p['id']][:,h]=1;p['done'][h]=True
                pending[:]=[p for p in pending if not p['done'][-1]]
            with Worker(result['backend'],a.out/'workers'/name) as w:
                for ri,item in enumerate(records):
                    q=item['request'];t=q['t'];finish(t,False)
                    if 'cloud' in q:
                        cloud=q['cloud'];w.request(dict(op='update',t=t,cloud=cloud));sequence.append(dict(t=t,cloud=cloud))
                        previous=latest;latest=load_cloud(cloud)
                        for p in pending:
                            schedules=[s for s in p['schedules'] if p['alive'][s['id']]]
                            if not schedules:continue
                            r=w.request(dict(op='label_schedules',t=t,schedules=schedules))
                            for b in r['branches']:
                                if not b['ok']:
                                    j=b['id'];p['alive'][j]=0;counts['new_map_invalidations']+=1
                                    # An observed first-failure event is NOT
                                    # right-censored just because the run aborts.
                                    for h,delta in enumerate(HORIZONS,1):
                                        if t<=p['t']+delta+1e-9:arrays['labels'][p['id']][j,h]=0;arrays['mask'][p['id']][j,h]=1
                                    details.append(dict(state=p['id'],candidate=j,observed_at=t,first_invalid_time=b['first_invalid_time'],lead=b['first_invalid_time']-t,map_stamp=r['map_stamp'],report=b))
                    finish(t,True)
                    if ri not in selected:continue
                    query={k:q[k] for k in ['t','start_state','goal','local_progress_target','map_context','execution_schedule'] if k in q};seed,features=candidate_bank(query)
                    query['local_progress_target']=query['map_context']['waypoint'];plans,_=decode(query,seed,np.column_stack([np.zeros(K),np.full(K,20.)]));mp=np.zeros((K,15),np.float32)
                    for bi in range(0,K,15):
                        mf=w.request(dict(op='candidate_features',t=t,direct_plans=plans[bi:bi+15]))
                        for e in mf['features']:mp[e['id']]=e['values']
                    image,history,state,feature,rule=inputs(latest,previous,query,result['backend'],seed,features,mp)
                    checked=dict(branches=[])
                    for bi in range(0,K,15):checked['branches']+=w.request(dict({k:v for k,v in query.items() if k!='execution_schedule'},op='evaluate_direct',direct_plans=plans[bi:bi+15],forbid_optimizer=True))['branches']
                    valid=[dict(id=b['id'],plan=b['plan']) for b in checked['branches'] if b['ok']];stops=[]
                    for bi in range(0,len(valid),15):stops+=w.request(dict(op='make_stops',t=t,direct_plans=valid[bi:bi+15]))['branches']
                    byid={b['id']:b['plan'] for b in stops if b['ok']};schedules=[]
                    label=np.zeros((K,4),np.float32);mask=np.zeros_like(label);mask[:,0]=1
                    for p in valid:
                        j=p['id'];pl=p['plan'];old=[c for c in query['execution_schedule'] if c['start']<pl['start']-1e-9];schedules.append(dict(id=j,execution_schedule=old+[pl,byid[j]]));label[j,0]=1
                    if schedules:
                        initial=w.request(dict(op='label_schedules',t=t,schedules=schedules))
                        for b in initial['branches']:
                            if not b['ok']:label[b['id'],0]=0;counts['initial_prefix_invalid']+=1
                    idx=len(rows);row=dict(index=idx,case=name,source_case=str(folder.resolve()),request_index=ri+1,backend=result['backend'],seed=result['seed'],split=result['split'],original_split=result.get('original_split',result['split']),policy=result['policy'],outcome=result['outcome'],query=query,cloud=cloud,previous_cloud=None if previous is None else {k:v for k,v in previous.items() if k not in ['points','depth_grid_world','depth_valid']},prefix=sequence.copy(),schedules=schedules,label_source='logged_future_received_maps')
                    rows.append(row)
                    for key,value in [('image',image),('previous',history),('state',state),('features',feature),('seed',seed),('labels',label),('mask',mask)]:arrays[key].append(value)
                    pending.append(dict(id=idx,t=t,alive=label[:,0].copy(),schedules=schedules,done=[True,False,False,False]))
                    for b in checked['branches']:counts[b['reason']]+=1
                finish(result['final_time'],True)
            print(json.dumps(dict(case=name,states=len(rows),invalidations=counts['new_map_invalidations'],seconds=time.perf_counter()-begin)),flush=True)
    data={k:np.asarray(v,np.float32) for k,v in arrays.items()};np.savez_compressed(a.out/'dataset.npz',**data);save(a.out/'records.json',rows);save(a.out/'future_invalidations.json',details)
    save(a.out/'protocol.json',dict(states=len(rows),train=sum(r['split']=='train' for r in rows),validation=sum(r['split']=='validation' for r in rows),sources=[str(p.resolve()) for p in a.runs],counts=dict(counts),horizons=HORIZONS,current_valid=int(data['labels'][:,:,0].sum()),future_negative=int(((data['labels'][:,:,0:1]>0)&(data['labels'][:,:,1:]<.5)&(data['mask'][:,:,1:]>0)).sum()),censored=int((data['mask'][:,:,1:]==0).sum()),dataset_sha256=hashlib.sha256((a.out/'dataset.npz').read_bytes()).hexdigest(),semantics='Cumulative validity of complete nominal/stop schedule on subsequently fused maps at +0.2/0.4/0.8 seconds. Future observations only label, never input. These labels initially follow the logged policy camera; candidate-specific rendering added separately. No claim of global map monotonicity or infinite-horizon safety.'))

if __name__=='__main__':main()
