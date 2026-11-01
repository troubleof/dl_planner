"""Real-worker rejection tests for backup continuity, expiry and changed map."""
import argparse,copy
from navigation_common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for backend in ['ego','rog']:
        with Worker(backend,a.out/backend,route_policy='repair') as w:
            sc=make_scene('empty');sensor=Sensor()
            for i in range(4):w.request(dict(op='update',t=.2*i,cloud=sensor.capture(sc,sc['start'],.2*i)))
            expected=stationary(sc['start']);r=w.request(dict(op='plan',t=1.,start_state=expected,goal=sc['goal'],with_backup=True))
            assert r['ok'] and 'backup_plan' in r
            q=dict(op='validate',t=1.05,plan=r['plan'],backup_plan=r['backup_plan'],start_state=expected)
            valid=w.request(q);assert valid['ok']
            altered=copy.deepcopy(q);altered['backup_plan']['controls'][0][0]+=.01
            bad=w.request(altered);assert not bad['ok'] and bad['reason']=='start_discontinuity'
            stale=w.request(dict(q,t=1.11));assert not stale['ok'] and stale['reason']=='stale_cloud'
            late=w.request(dict(q,t=r['plan']['start']-.025));assert not late['ok'] and late['reason']=='deadline'
            curve=Curve(r['backup_plan']);end=curve.state(curve.start+curve.duration)
            # Inject observed obstacle at the stored stop endpoint, with a real
            # sensor cloud. It must invalidate the composite schedule.
            x=end['p'][0]
            changed=dict(sc,obstacles=[dict(kind='box',lo=[x-.15,-2.,0.],hi=[x+.15,2.,5.])])
            guarded=w.request(dict(op='guard',t=1.2,cloud=sensor.capture(changed,sc['start'],1.2),
                execution_schedule=[r['plan'],r['backup_plan']],nominal_plan=r['plan'],find_stop=True))
            assert not guarded['ok'] and guarded['first_invalid_time'] is not None
            rows.append(dict(backend=backend,valid_bundle=True,discontinuous_backup_rejected=True,stale_rejected=True,deadline_rejected=True,
                             changed_map_invalidates_backup=True,changed_map_report=guarded))
            print(backend,'5 backup contract checks passed',flush=True)
    save(a.out/'summary.json',rows)

if __name__=='__main__':main()
