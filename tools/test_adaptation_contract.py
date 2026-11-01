import argparse,json
from pathlib import Path
import numpy as np
from navigation_common import *
from adapted_frontends import inputs,primitives,AlignedYopo,RangeFrontend

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--models',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    scene=make_scene('empty');sensor=Sensor();cloud=sensor.capture(scene,scene['start'],.8);q=dict(op='plan',t=1.,start_state=stationary(scene['start']),goal=scene['goal'],with_backup=True)
    image,state=inputs(cloud,q);assert image.shape==(3,17,72) and state.shape==(22,) and np.all(image[0]<=1.)
    rows=[]
    for backend in ['ego','rog']:
        with Worker(backend,a.out/backend) as w:
            for i in range(5):w.request(dict(op='update',t=i*.2,cloud=sensor.capture(scene,scene['start'],i*.2)))
            before=w.request(q);assert before['ok']
            evaluated=w.request(dict(q,op='evaluate_routes',candidates=primitives(q)))
            assert len(evaluated['branches'])==16 and evaluated['branches'][15]['ok']
            after=w.request(q);assert after['ok']
            assert np.max(np.abs(np.array(before['plan']['controls'])-after['plan']['controls']))<1e-8,'offline branches changed repair state'
            bad=primitives(q);bad[0]['points'][0][0]+=1
            rejected=w.request(dict(q,neural_routes=[bad[0]],probe_only=True));assert not rejected['ok'] and rejected['reason']=='reference_start'
            validated=w.request(dict(op='validate',t=1.01,plan=after['plan'],backup_plan=after['backup_plan'],start_state=q['start_state']));assert validated['ok']
            late=w.request(dict(op='validate',t=1.30,plan=after['plan'],start_state=q['start_state']));assert not late['ok'] and late['reason']=='deadline'
            rows.append(dict(backend=backend,branches=len(evaluated['branches']),valid=sum(r['ok'] for r in evaluated['branches']),cache_preserved=True,deadline_preserved=True,invalid_geometry_refused=True))
    frontends=[AlignedYopo(),RangeFrontend()]
    if a.models:frontends += [RangeFrontend(ROOT/'models'/f'range_scorer_s{s}.pt') for s in [11,22,33]]
    for idx,f in enumerate(frontends):
        routes,info=f.propose(cloud,q,a.out/f'f{idx}',1);assert len(routes)<=3
        for route in routes:assert np.linalg.norm(np.array(route['points'][0])-q['start_state']['p'])<1e-8
        routes,info=f.propose(cloud,dict(q,t=1.31),a.out/f'f{idx}',2);assert not routes and info['reason']=='stale_received_cloud'
    save(a.out/'summary.json',dict(checks=rows,frontends=len(frontends),native_shape=list(image.shape),state_shape=list(state.shape)))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
