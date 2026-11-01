"""Motion direction, no-return, frame alignment, strict rejection and deadline checks."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from navigation_common import Sensor,Worker,make_scene,stationary,save,RADIUS
from motion_frontend import observation,directional

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    points=np.array([[0,1,2.5],[1.5,0,2.5]])
    c,acc,m=directional(points,[0,0,2.5],[2,0,0],[1,0,0]);assert m[1]<m[0] and c[0]==0 and c[1]==2
    _,_,away=directional(points,[0,0,2.5],[-2,0,0],[-1,0,0]);assert away[1]>m[1]
    _,_,stationary_acc=directional(points,[0,0,2.5],[0,0,0],[2,0,0]);assert stationary_acc[1]<1.5-RADIUS
    empty=make_scene('empty');cloud=Sensor().capture(empty,empty['start'],.8);q=dict(t=1.,start_state=stationary(empty['start']),goal=empty['goal'])
    image,_=observation(cloud,q,'ego');assert image.shape==(6,17,72) and np.all(image[3:]==0)
    scene=make_scene('offset_wall');cl=Sensor().capture(scene,scene['start'],.8);q['start_state'].update(v=[2,0,0],a=[1,0,0]);im,st=observation(cl,q,'ego')
    shift=np.array([1.25,-.7,0]);moved=copy.deepcopy(cl);moved['origin']=(np.asarray(cl['origin'])+shift).tolist();moved['points']=(np.asarray(cl['points'])+shift).tolist()
    mq=copy.deepcopy(q);mq['start_state']['p']=(np.asarray(q['start_state']['p'])+shift).tolist();mq['goal']=(np.asarray(q['goal'])+shift).tolist()
    im2,st2=observation(moved,mq,'ego');assert np.allclose(im,im2,atol=2e-6) and np.allclose(st,st2,atol=2e-6)
    results=[]
    for backend in ['ego','rog']:
        with Worker(backend,a.out/backend,route_policy='repair') as w:
            for i in range(4):w.request(dict(op='update',t=i*.2,cloud=Sensor().capture(empty,empty['start'],i*.2)))
            p=np.asarray(empty['start']);controls=np.repeat(p[None],12,axis=0);controls[:,0]+=[0,0,0,.12,.35,.7,1.05,1.3,1.45,1.5,1.5,1.5]
            plan=dict(start=1.3,dt=1/3,duration=3.,controls=controls.tolist());query=dict(op='plan',t=1.,start_state=stationary(p),goal=empty['goal'],direct_plans=[dict(id=0,plan=plan)],forbid_optimizer=True,with_backup=True)
            w.plan_budget_ms=0.;late=w.request(query);assert not late['ok'] and late['plan_budget_exceeded'] and late['optimizer_calls']==0 and late.get('direct_tried',0)==0
            w.plan_budget_ms=1000.;valid=w.request(query);assert valid['ok'] and valid['optimizer_calls']==0
            rec=w.request(dict(query,op='evaluate_recovery'));assert len(rec['branches'][0]['stops'])==6
            results.append(dict(backend=backend,late=late['reason'],valid=True,recovery=rec))
    save(a.out/'summary.json',dict(passed=True,closer_lateral_margin=float(m[0]),farther_forward_margin=float(m[1]),checks=['direction','acceleration_from_rest','no_return','world_translation','zero_budget_reject','same_validator','offline_stops'],backends=results))
    print('motion contracts passed',flush=True)

if __name__=='__main__':main()
