"""Re-evaluate actual candidate bundles at identical causal prefixes, then repair for 6s."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from backup_trial import trial,conditions
from navigation_common import ROOT,Curve,save
from recovery_frontend import RecoveryFrontend,ACTIONS
from branch_resume import PrefixMismatch

HORIZON=6.


def measure(folder,result,goal):
    trajectory=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1)
    log=[json.loads(s) for s in (folder/'worker/requests.jsonl').read_text().splitlines()]
    first=next(x for x in log if x['request']['op']=='plan' and not x['request'].get('_skip_frontend',False))
    q,r=first['request'],first['response']
    assert abs(q['t']-result['initial_time'])<1e-8
    events=json.loads((folder/'events.json').read_text());timings=json.loads((folder/'timings.json').read_text())
    bundles=json.loads((folder/'plans.json').read_text())
    committed=any(abs(p['nominal']['start']-r.get('plan',{}).get('start',-100))<1e-8 for p in bundles)
    distance=np.linalg.norm(trajectory[:,1:4]-goal,axis=1)
    progress=(distance[0]-distance[-1])/HORIZON
    failed=result['outcome'] not in ['reached','branch_horizon']
    last=trajectory[:,0]>=trajectory[-1,0]-2.
    stalled=not failed and result['outcome']!='reached' and distance[-1]>.5 and distance[last][0]-distance[-1]<.15
    cost=sum(t['rpc_ms'] for t in timings)/1000
    early=(HORIZON-min(result['duration_s'],HORIZON))/HORIZON if result['outcome']=='reached' else 0.
    long=[float(progress),float(failed),float(stalled),cost,float(committed),early]
    if r.get('ok'):
        curve=Curve(r['plan']);start=np.asarray(q['start_state']['p'])
        short_progress=(np.linalg.norm(start-goal)-np.linalg.norm(np.asarray(curve.state(curve.start+.6)['p'])-goal))/.6
    else:short_progress=0.
    short=[float(short_progress),float(not r.get('ok')),0.,r['rpc_ms']/1000,float(r.get('ok')),0.]
    return long,short,dict(initial_request=q,initial_response=r,first_bundle_committed=committed,
        first_proposal_committed=committed and r.get('frontend')=='yopo',
        failure_checks=[e for e in events if e['kind']=='execution_check' and not e['report']['ok']],
        last_two_seconds_progress=float(distance[last][0]-distance[-1]),last_speed=result['final_speed'])


def select(source,hard,count):
    log=[json.loads(x) for x in (source/'worker/requests.jsonl').read_text().splitlines()]
    eligible=[i for i,x in enumerate(log) if x['request']['op']=='plan' and np.linalg.norm(np.asarray(x['request']['goal'])-x['request']['start_state']['p'])>=3.]
    outcome=json.loads((source/'result.json').read_text())['outcome']
    if hard and outcome!='reached':
        selected=[eligible[max(0,len(eligible)-k)] for k in [6,4,2]]
    else:
        selected=[eligible[i] for i in np.unique(np.rint(np.linspace(1,max(1,len(eligible)-2),min(count,len(eligible)-1))).astype(int))]
    return sorted(set(selected)),log


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--old-adaptation',type=Path,required=True);ap.add_argument('--old-trajectory',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--midpoints',action='store_true')
    ap.add_argument('--resume-from',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    sources=[]
    for row in json.loads((a.old_adaptation/'runs/train-rollouts-v1/summary.json').read_text()):
        sources.append(dict(folder=a.old_adaptation/'runs/train-rollouts-v1'/row['folder'],backend=row['backend'],seed=row['seed'],split=row['split'],hard=False,count=2))
    old=a.old_trajectory/'runs/trajectory-test-v1'
    for row in json.loads((old/'summary.json').read_text()):
        if (row['seed'] in [3503,3504] and row['policy'] in ['trajectory11','trajectory33']) or row['folder']=='offset_wall_0_rog_old11':
            sources.append(dict(folder=old/row['folder'],backend=row['backend'],seed=row['seed'],split='train',hard=True,count=3))
    if a.pilot:sources=[next(s for s in sources if s['folder'].name=='forest_3503_rog_trajectory33')]
    if a.midpoints:sources=[s for s in sources if not s['hard']]
    records=[];arrays={k:[] for k in ['image','state','candidates','long','short']};restoration_failures=[];skipped=[];skip_first_unfinished=False
    if a.resume_from:
        import os
        records=json.loads((a.resume_from/'records.json').read_text());prior=np.load(a.resume_from/'dataset.npz')
        arrays={k:list(prior[k]) for k in arrays}
        for record in records:
            for branch in record['branches']:
                branch['folder']=os.path.relpath(a.resume_from/branch['folder'],a.out)
        restoration_failures=json.loads((a.resume_from/'restoration_failures.json').read_text())
        assert len(restoration_failures)>=3 and all(f['before_branch'] for f in restoration_failures[-3:])
        skip_first_unfinished=True
    completed={(r['source_folder'],r['source_request_index']) for r in records}
    save(a.out/'source_manifest.json',dict(horizon=HORIZON,midpoints=a.midpoints,collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=[dict(s,folder=str(s['folder'])) for s in sources],
        old_test_reclassified_as_development=[3503,3504,'offset_wall'],new_test_seeds=[3601,3602,3603,3604]))
    for spec in sources:
        folder=spec['folder'];indices,log=select(folder,spec['hard'],spec['count'])
        if a.midpoints:
            eligible=[i for i,x in enumerate(log) if x['request']['op']=='plan' and np.linalg.norm(np.asarray(x['request']['goal'])-x['request']['start_state']['p'])>=3.]
            indices=[eligible[len(eligible)//2]]
        if a.pilot:indices=indices[-1:]
        scene=json.loads((folder/'scene.json').read_text())
        for index in indices:
            if (str(folder),index) in completed:continue
            if skip_first_unfinished:
                skipped.append(dict(source_folder=str(folder),source_request_index=index,reason='three prior restoration failures before any experimental branch',prior_run=str(a.resume_from)))
                save(a.out/'skipped_initializations.json',skipped);skip_first_unfinished=False;continue
            state_id=len(records);branches=[];values=[];short_values=[];observations=[]
            skip_state=False
            state_key=hashlib.sha256((str(folder)+':'+str(index)).encode()).hexdigest()[:8]
            for action in range(4):
                for attempt in range(3):
                    output=a.out/'branches'/f's{state_id:03d}_{state_key}_a{action}_init{attempt}'
                    try:
                        result=trial(spec['backend'],scene,conditions()[0],12000,output,max_time=HORIZON,
                            frontend=RecoveryFrontend(spec['backend'],mode='forced',forced_action=action),resume_from=(folder,index))
                        break
                    except PrefixMismatch as error:
                        failure=dict(folder=str(output.relative_to(a.out)),reason=str(error),before_branch=True)
                        restoration_failures.append(failure);save(output/'restoration_failure.json',failure)
                        save(a.out/'restoration_failures.json',restoration_failures)
                        if attempt==2:skip_state=True;break
                if skip_state:break
                long,short,diagnostic=measure(output,result,np.asarray(scene['goal']))
                save(output/'labels.json',dict(long=long,short=short,diagnostic=diagnostic))
                neural=next((output/'worker/neural').glob('*.npz'))
                data=np.load(neural);observations.append({k:data[k] for k in ['range_image','state','candidates']})
                assert np.max(np.abs(np.asarray(diagnostic['initial_request']['start_state']['p'])-log[index]['request']['start_state']['p']))<1e-8
                assert diagnostic['initial_request']['neural_input']['action']==action
                if action:
                    for key in observations[0]:assert np.array_equal(observations[0][key],observations[action][key]),key
                branches.append(dict(action=action,folder=str(output.relative_to(a.out)),result=result))
                values.append(long);short_values.append(short)
            if skip_state:
                skipped.append(dict(source_folder=str(folder),source_request_index=index,reason='three restoration failures before comparable branch set',partial_branches=branches))
                save(a.out/'skipped_initializations.json',skipped);continue
            arrays['image'].append(observations[0]['range_image']);arrays['state'].append(observations[0]['state']);arrays['candidates'].append(observations[0]['candidates'])
            arrays['long'].append(values);arrays['short'].append(short_values)
            records.append(dict(state_id=state_id,source_folder=str(folder),source_request_index=index,
                source_t=log[index]['request']['t'],source_sha256=hashlib.sha256((folder/'worker/requests.jsonl').read_bytes()).hexdigest(),
                seed=spec['seed'],backend=spec['backend'],split=spec['split'],hard=spec['hard'],branches=branches))
            save(a.out/'records.json',records);np.savez_compressed(a.out/'dataset.npz',**{k:np.asarray(v,np.float32) for k,v in arrays.items()})
            print(json.dumps(dict(states=len(records),source=folder.name,t=records[-1]['source_t'],outcomes=[b['result']['outcome'] for b in branches],fail=[v[1] for v in values],stall=[v[2] for v in values])),flush=True)
    save(a.out/'protocol.json',dict(states=len(records),branches=4*len(records),horizon_s=HORIZON,actions=ACTIONS,
        continuation='repair immediately after the first request; branch_horizon is not mission success',
        restoration_failures=restoration_failures,skipped_initializations=skipped,same_input_arrays_verified=True,
        dataset_sha256=hashlib.sha256((a.out/'dataset.npz').read_bytes()).hexdigest()))


if __name__=='__main__':main()
