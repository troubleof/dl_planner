#!/usr/bin/env python3
"""Exercise route-cache invalidation through real point-cloud map adapters."""
import argparse
from pathlib import Path
from navigation_common import *

def main(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    results=[]
    for backend in ['ego','rog']:
        scene=make_scene('empty');scene['start']=[-3.,0.,2.5];scene['goal']=[3.,0.,2.5]
        sensor=Sensor()
        with Worker(backend,Path(out)/backend,route_policy='repair') as w:
            for i in range(4):w.request(dict(op='update',t=i*.2,cloud=sensor.capture(scene,scene['start'],i*.2)))
            s=stationary(scene['start'])
            old=w.request(dict(op='plan',t=.8,start_state=s,goal=scene['goal']))
            assert old['ok'],old
            scene['obstacles']=[dict(kind='box',lo=[-1.55,-.45,0.],hi=[-1.25,.45,5.])]
            # Two fresh observations are supplied before the original deadline.
            # This is a map invalidation test, not a 5 Hz flight experiment.
            for stamp in [.82,.84]:w.request(dict(op='update',t=stamp,cloud=sensor.capture(scene,scene['start'],stamp)))
            check=w.request(dict(op='validate',t=.86,plan=old['plan'],start_state=s))
            assert not check['ok'] and check['reason']=='occupied',check
            new=w.request(dict(op='plan',t=.88,start_state=s,goal=scene['goal']))
            assert new.get('route_source') in ['repair','full'],new
            fallback='not_needed'
            if not new['ok']:
                # A repaired geometric route is not an executable trajectory.
                # Sparse observations may still make its spline unacceptable.
                assert new['reason'] in ['unknown','occupied','refine_failed','rebound_failed','no_known_route','derivative_bound'],new
                retry=w.request(dict(op='plan',t=.90,start_state=s,goal=scene['goal']))
                assert retry['route_source']=='full',retry
                fallback=retry['route_fallback']
            result=dict(backend=backend,old_candidate_rejected=check['reason'],replacement_source=new['route_source'],replacement_valid=new['ok'],replacement_reason=new['reason'],local_repairs=new.get('local_repairs',0),retry_fallback=fallback)
        # Independently test a valid reused candidate arriving after its deadline.
        empty=make_scene('empty');empty['start']=scene['start'];empty['goal']=scene['goal']
        with Worker(backend,Path(out)/(backend+'_late'),route_policy='repair') as w:
            for i in range(4):w.request(dict(op='update',t=i*.2,cloud=sensor.capture(empty,empty['start'],i*.2)))
            initial=w.request(dict(op='plan',t=.8,start_state=s,goal=empty['goal']));assert initial['ok'],initial
            reused=w.request(dict(op='plan',t=.82,start_state=s,goal=empty['goal']));assert reused['ok'] and reused['route_source']=='reuse',reused
            late=w.request(dict(op='validate',t=reused['plan']['start']-.024,plan=reused['plan'],start_state=s))
            assert not late['ok'] and late['reason']=='deadline',late
            result['late_candidate_rejected']=late['reason']
            results.append(result);print(json.dumps(result),flush=True)
    save(Path(out)/'summary.json',results)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
