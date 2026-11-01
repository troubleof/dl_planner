"""Causal prefix forks; five closed-loop actions with a recorded continuation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from backup_trial import trial,conditions
from navigation_common import ROOT,Curve,save
from context_frontend import ACTION_IDS,ContextFrontend
from branch_resume import PrefixMismatch

def labels(folder,result,goal):
    traj=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1);timing=json.loads((folder/'timings.json').read_text());plans=json.loads((folder/'plans.json').read_text())
    req=[r for r in timing if r['op']=='plan'];first=req[0];records=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()]
    actual=next(x for x in records if x['request']['op']=='plan' and x['request']['t']>=result['initial_time']-1e-9 and not x['request'].get('_skip_frontend',False))
    failure=result['outcome'] not in ['branch_horizon','reached'];progress=(np.linalg.norm(traj[0,1:4]-goal)-np.linalg.norm(traj[-1,1:4]-goal))/2.
    committed=any(abs(p['nominal']['start']-actual['response'].get('plan',{}).get('start',-100))<1e-8 for p in plans)
    multi=[float(progress),float(failure),sum(r['rpc_ms'] for r in timing)/100.,float(result['backup_activations']),float(committed)]
    if actual['response'].get('ok'):
        curve=Curve(actual['response']['plan']);p=np.array(actual['request']['start_state']['p']);progress_short=(np.linalg.norm(p-goal)-np.linalg.norm(np.array(curve.state(curve.start+.6)['p'])-goal))/.6
        short=[float(progress_short),0.,first['rpc_ms']/100.,0.,1.]
    else:short=[0.,1.,first['rpc_ms']/100.,0.,0.]
    return multi,short

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--states-per-case',type=int,default=8);ap.add_argument('--continuation-weight',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((a.run/'summary.json').read_text());data={k:[] for k in ['rays','local_map','state','multistep','short']};records=[];setup_failures=[]
    for row in rows:
        folder=a.run/row['folder'];log=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()];scene=json.loads((folder/'scene.json').read_text())
        eligible=[i for i,x in enumerate(log) if x['request']['op']=='plan' and x['request'].get('neural_input',{}).get('reason')=='ok' and np.linalg.norm(np.array(x['request']['goal'])-x['request']['start_state']['p'])>=3.]
        chosen=[eligible[i] for i in np.unique(np.rint(np.linspace(0,len(eligible)-1,min(a.states_per_case,len(eligible)))).astype(int))] if eligible else []
        for index in chosen:
            npz=np.load(folder/'worker/neural'/f'{index+1:05d}.npz');multi=[];short=[];branches=[]
            for action in range(5):
                for attempt in range(3):
                    output=a.out/'branches'/(f's{len(records):04d}_a{action}'+(f'_restore{attempt}' if attempt else ''))
                    try:
                        r=trial(row['backend'],scene,conditions()[0],12000,output,max_time=2.,frontend=ContextFrontend([a.continuation_weight] if a.continuation_weight else None,mode='forced',forced_action=action),resume_from=(folder,index))
                        break
                    except PrefixMismatch as error:
                        failure=dict(folder=str(output.relative_to(a.out)),reason=str(error),phase='prefix_restoration_before_branch')
                        setup_failures.append(failure);save(output/'prefix_setup_failure.json',failure);save(a.out/'setup_failures.json',setup_failures)
                        if attempt==2:raise
                m,s=labels(output,r,np.array(scene['goal']));multi.append(m);short.append(s);branches.append(dict(folder=str(output.relative_to(a.out)),action=action,candidate_id=ACTION_IDS[action],result=r))
            for k in ['rays','local_map','state']:data[k].append(npz[k])
            data['multistep'].append(multi);data['short'].append(short)
            records.append(dict(source_folder=str(folder),source_request_index=index,backend=row['backend'],seed=row['seed'],split=row['split'],stamp=log[index]['request']['neural_input']['capture_stamp'],t=log[index]['request']['t'],branches=branches))
            save(a.out/'records.json',records)
            print(json.dumps(dict(case=row['folder'],states=len(records),source_time=log[index]['request']['t'],outcomes=[b['result']['outcome'] for b in branches])),flush=True)
        np.savez_compressed(a.out/'dataset.npz',**{k:np.asarray(v,dtype=np.float32) for k,v in data.items()})
    save(a.out/'protocol.json',dict(states=len(records),branches=len(records)*5,action_ids=ACTION_IDS,horizon_s=2.,continuation=str(a.continuation_weight) if a.continuation_weight else 'repair with identical context pipeline after first action',
        actual_simulation_timing=True,source_run=str(a.run),prefix_setup_failures=setup_failures,dataset_sha256=hashlib.sha256((a.out/'dataset.npz').read_bytes()).hexdigest(),limits='finite action set and 2-second improvement oracle; not globally optimal labels'))
if __name__=='__main__':main()
