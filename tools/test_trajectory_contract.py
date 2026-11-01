import argparse
import copy
from pathlib import Path
import numpy as np
from navigation_common import *
from trajectory_frontend import TrajectoryFrontend,decode


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    scene=make_scene('empty');sensor=Sensor();cloud=sensor.capture(scene,scene['start'],.8)
    q=dict(op='plan',t=1.,start_state=stationary(scene['start']),goal=scene['goal'],with_backup=True)
    results=[]
    for seed in [11,22,33]:
        model=TrajectoryFrontend(ROOT/'trajectory_models'/f'trajectory_s{seed}.pt')
        routes,info=model.propose(cloud,q,a.out/f'seed{seed}',1)
        assert info['reason']=='ok' and len(routes)<=3
        for r in routes:
            assert r['points'][0]==q['start_state']['p'] and len(r['points'])==4
        for index,(bad,request,why) in enumerate([
            (None,q,'no_received_cloud'),(cloud,dict(q,t=1.31),'invalid_cloud_age'),
            (cloud,dict(q,t=.79),'invalid_cloud_age'),
            (dict(cloud,points=cloud['points'][:-1]),q,'unsupported_or_nonfinite_cloud'),
            (cloud,dict(q,goal=[-6.,0.,2.5]),'near_goal_repair')],2):
            chosen,meta=model.propose(bad,request,a.out/f'seed{seed}',index)
            assert not chosen and meta['reason']==why
        bad=copy.deepcopy(cloud);bad['points'][0][0]=float('nan')
        assert model.propose(bad,q,a.out/f'seed{seed}',9)[1]['reason']=='unsupported_or_nonfinite_cloud'
        results.append(dict(seed=seed,parameters=model.metadata['parameters'],proposals=len(routes),fallbacks_passed=True))
    # Worker checks are independent of the trained model's selected action.
    scores=np.zeros(16);scores[7]=10.;scores[15]=1.
    forced,_=decode(q,scores,np.ones(16),np.zeros((15,3,3)))
    assert len(forced)==1 and forced[0]['id']==7
    for backend in ['ego','rog']:
        with Worker(backend,a.out/backend,route_policy='repair') as w:
            for i in range(5):w.request(dict(op='update',t=i*.2,cloud=sensor.capture(scene,scene['start'],i*.2)))
            r=w.request(dict(q,neural_routes=forced));assert r['ok'] and r['frontend']=='yopo'
            valid=w.request(dict(op='validate',t=1.01,plan=r['plan'],backup_plan=r['backup_plan'],start_state=q['start_state']))
            assert valid['ok']
            late=w.request(dict(op='validate',t=1.30,plan=r['plan'],start_state=q['start_state']))
            assert not late['ok'] and late['reason']=='deadline'
            malformed=copy.deepcopy(forced);malformed[0]['points'][0][0]+=1.
            fallback=w.request(dict(q,neural_routes=malformed))
            assert fallback['ok'] and fallback['frontend']=='repair' and fallback['neural_geometric_rejects']==1
            assert fallback['neural_last_rejection']=='reference_start'
            results.append(dict(backend=backend,optimizer_and_backup=True,deadline_enforced=True,malformed_reference_falls_back=True))
    save(a.out/'summary.json',dict(passed=True,checks=results));print(json.dumps(results),flush=True)


if __name__=='__main__':main()
