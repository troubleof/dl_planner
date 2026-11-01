import argparse,json
from pathlib import Path
import numpy as np
from navigation_common import *
from backup_trial import trial,conditions
from context_frontend import ContextFrontend,features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);checks=[]
    for backend in ['ego','rog']:
        with Worker(backend,a.out/backend,frontend=ContextFrontend(mode='repair')) as w:
            origin=[-7.,0.,2.5]
            w.request(dict(op='update',t=0.,cloud=dict(frame='world',stamp=0.,origin=origin,points=[[-6.,0.,2.5]])))
            w.request(dict(op='update',t=.2,cloud=dict(frame='world',stamp=.2,origin=origin,points=[[-7.,1.,2.5]])))
            age=w.request(dict(op='observation_query',t=.2,points=[[-6.45,.05,2.55],[-6.95,.55,2.55],[-6.45,.55,2.55]]))['stamps']
            assert age==[0.,.2,-1.],age
            checks.append(dict(backend=backend,independent_voxel_timestamps=age))
    # Fresh contract source episode, then five counterfactual actions from the
    # exact same pending-plan point. Source future observations cannot be reused.
    source=a.out/'source';scene=make_scene('empty');trial('ego',scene,conditions()[0],12000,source,max_time=4.,frontend=ContextFrontend(mode='repair'))
    log=[json.loads(x) for x in (source/'worker/requests.jsonl').read_text().splitlines()]
    index=next(i for i,x in enumerate(log) if x['request']['op']=='plan' and x['request']['t']>=2.)
    target=log[index]['request'];original=np.loadtxt(source/'trajectory.csv',delimiter=',',skiprows=1);start=original[np.argmin(abs(original[:,0]-target['t']))]
    ends=[]
    for action in range(5):
        folder=a.out/f'branch{action}';r=trial('ego',scene,conditions()[0],12000,folder,max_time=2.,frontend=ContextFrontend(mode='forced',forced_action=action),resume_from=(source,index))
        traj=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1);assert np.max(np.abs(traj[0,:10]-start[:10]))<1e-8
        entries=[json.loads(s) for s in (folder/'worker/requests.jsonl').read_text().splitlines()]
        first=next((i,x) for i,x in enumerate(entries,1) if x['request']['op']=='plan' and not x['request'].get('_skip_frontend'))
        q=first[1]['request'];assert q['neural_input']['action']==action
        expected=np.load(source/'worker/neural'/f'{index+1:05d}.npz');actual=np.load(folder/'worker/neural'/f'{first[0]:05d}.npz')
        assert np.array_equal(expected['rays'],actual['rays']) and np.array_equal(expected['local_map'],actual['local_map'])
        for x in entries:
            q=x['request'];cloud=q.get('cloud')
            if cloud:
                assert cloud['stamp']<=q['t']
                if cloud['stamp']>traj[0,0]+1e-9:
                    sample=traj[np.argmin(abs(traj[:,0]-cloud['stamp']))]
                    assert np.max(abs(np.array(cloud['origin'])-sample[1:4]))<1e-8
        ends.append(traj[-1,1:4].tolist());checks.append(dict(action=action,outcome=r['outcome'],same_initial_pva=True,same_observation=True,future_rerendered=True))
    assert np.linalg.norm(np.array(ends[0])-ends[2])>.1,'counterfactual actions did not diverge'
    save(a.out/'summary.json',dict(checks=checks,final_positions=ends));print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
