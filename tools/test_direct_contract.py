import argparse,copy
from pathlib import Path
import numpy as np
from navigation_common import Worker,Sensor,make_scene,stationary,Curve,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);results=[]
    for backend in ['ego','rog']:
        scene=make_scene('empty');p=np.asarray(scene['start']);state=stationary(p);x=np.array([0,0,0,.12,.35,.7,1.05,1.3,1.45,1.5,1.5,1.5])
        controls=np.repeat(p[None],12,axis=0);controls[:,0]+=x
        plan=dict(start=1.3,dt=1/3,duration=3.,controls=controls.tolist())
        q=dict(op='plan',t=1.,start_state=state,goal=scene['goal'],direct_plans=[dict(id=0,plan=plan)],with_backup=True,forbid_optimizer=True)
        with Worker(backend,a.out/backend,route_policy='repair') as w:
            sensor=Sensor()
            for i in range(4):w.request(dict(op='update',t=i*.2,cloud=sensor.capture(scene,p,i*.2)))
            r=w.request(q);assert r['ok'] and r['frontend']=='direct' and r['optimizer_calls']==0 and 'backup_plan' in r
            for key in plan:assert np.allclose(plan[key],r['plan'][key],atol=1e-12,rtol=0)
            cases={}
            def reject(name,mutate,expected):
                test=copy.deepcopy(q);mutate(test);res=w.request(test)
                assert not res['ok'] and res['optimizer_calls']==0
                if expected:assert res['direct_checks'][0]['reason']==expected,(name,res)
                cases[name]=res
            reject('absolute_start',lambda z:z['direct_plans'][0]['plan'].update(start=1.31),'direct_start_time')
            reject('duration_metadata',lambda z:z['direct_plans'][0]['plan'].update(duration=3.1),'direct_duration')
            reject('pva',lambda z:z['start_state']['v'].__setitem__(0,.1),'start_discontinuity')
            reject('terminal_rest',lambda z:z['direct_plans'][0]['plan']['controls'][-1].__setitem__(0,-5.49),'terminal_not_rest')
            reject('dynamic_bound',lambda z:z['direct_plans'][0]['plan']['controls'][5].__setitem__(0,-2.),'derivative_bound')
            def retreat(z):z['direct_plans'][0]['plan']['controls']=(np.repeat(p[None],12,axis=0)-np.column_stack([x,np.zeros((12,2))])).tolist()
            reject('nonprogressing_goal',retreat,'no_terminal_progress')
            def stale(z):z['t']=1.2;z['direct_plans'][0]['plan']['start']=1.5
            reject('stale',stale,'stale_cloud')
            def unknown(z):z['start_state']=stationary([7,0,2.5]);z['direct_plans'][0]['plan']['controls']=[[7,0,2.5]]*12
            reject('unknown',unknown,'unknown')
            reject('empty',lambda z:z.update(direct_plans=[]),None)
            safe=Curve(r['plan']);stop=Curve(r['backup_plan'])
            for k in ['p','v','a']:assert np.linalg.norm(np.asarray(safe.state(stop.start)[k])-stop.state(stop.start)[k])<1e-8
            results.append(dict(backend=backend,passed=True,cases=cases,valid_plan=r))
    save(a.out/'summary.json',dict(passed=True,backends=results));print('direct contracts passed',flush=True)

if __name__=='__main__':main()
