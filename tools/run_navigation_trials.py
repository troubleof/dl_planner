#!/usr/bin/env python3
"""Causal single-UAV experiments. Map, solve, recheck and IPC costs delay execution."""
import argparse,json,time
from pathlib import Path
import numpy as np
from navigation_common import *

def trial(backend,scene,extra_delay,out,max_time=35.,route_policy=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    effective_config=dict(CONFIG,route_policy=route_policy or CONFIG.get('route_policy','full'))
    save(out/'config.json',dict(effective_config,backend=backend,extra_delay_s=extra_delay))
    save(out/'scene.json',scene)
    sensor=Sensor();events=[];trajectory=[];rpc=[];generation=[];plans=[]
    initial=stationary(scene['start']);active=None;pending=None;pipeline=None
    t=1.;next_scan=t;next_plan=t;latest=None;map_stamp=.6;map_version=4
    reason='timeout';failures={};accepted=0;peak_v=0.;peak_a=0.;min_gap=100.
    start_wall=time.perf_counter()
    def state_at(at_t):return active.state(at_t) if active else initial
    def req(w,request):
        r=w.request(request);rpc.append(dict(t=request['t'],op=request['op'],**{k:v for k,v in r.items() if k not in ['plan','route','cells']}))
        return r
    def cloud_if_new(request):
        if latest is not None and latest['stamp']>map_stamp:request['cloud']=latest
        if active is not None:request['executing_plan']=active.plan
        return request
    def record_failure(why):failures[why]=failures.get(why,0)+1
    with Worker(backend,out/'worker',route_policy=route_policy) as w:
        for i in range(4):
            cloud=sensor.capture(scene,scene['start'],i*.2)
            req(w,dict(op='update',t=i*.2,cloud=cloud))
        while t<=1.+max_time+1e-9:
            if pending is not None and t>=pending.start-1e-9:
                before=state_at(pending.start);after=pending.state(pending.start)
                jump={k:float(np.linalg.norm(np.asarray(before[k])-after[k])) for k in ['p','v','a']}
                events.append(dict(kind='activate',t=t,jump=jump))
                if jump['p']>1e-6 or jump['v']>1e-6 or jump['a']>1e-5:reason='discontinuous_handoff';break
                active=pending;pending=None
            current=state_at(t);p=np.asarray(current['p']);v=np.asarray(current['v']);a=np.asarray(current['a'])
            gap=float(clearance(scene,p)-RADIUS);min_gap=min(min_gap,gap);peak_v=max(peak_v,float(np.linalg.norm(v)));peak_a=max(peak_a,float(np.linalg.norm(a)))
            trajectory.append([t,*p,*v,*a,gap])
            if gap<0:reason='envelope_collision';break
            if not CONFIG['height_band'][0]-1e-8<=p[2]<=CONFIG['height_band'][1]+1e-8:reason='height_violation';break
            if np.linalg.norm(p-scene['goal'])<.18 and np.linalg.norm(v)<.15:reason='reached';break
            if t>=next_scan-1e-9:
                begin=time.perf_counter();latest=sensor.capture(scene,p,t);generation.append((time.perf_counter()-begin)*1000)
                next_scan=round_tick(t+1/CONFIG['sensor_rate'])
            if pipeline is not None and t>=pipeline['ready']-1e-9:
                phase=pipeline['phase'];r=pipeline['result']
                if phase=='solve':
                    if r.get('reason')=='unsafe_execution_prefix':reason='unsafe_execution_prefix';break
                    if not r['ok']:
                        record_failure(r['reason']);events.append(dict(kind='solve_rejected',t=t,reason=r['reason']));pipeline=None;next_plan=t+.1
                    else:
                        request=cloud_if_new(dict(op='validate',t=t,plan=r['plan'],start_state=pipeline['expected']))
                        validated=req(w,request);map_stamp=validated['map_stamp'];map_version=validated['map_version']
                        pipeline=dict(phase='validate',ready=round_tick(t+validated['rpc_ms']/1000+CONFIG['publication_delay']+CONFIG['dt']),result=validated,plan=r['plan'],expected=pipeline['expected'],checked_version=map_version,started=pipeline['started'],route_source=r.get('route_source','full'))
                else:
                    begin=time.perf_counter();candidate=pipeline['plan'];why=r.get('reason','invalid')
                    ok=r['ok']
                    if why=='unsafe_execution_prefix':reason='unsafe_execution_prefix';break
                    if ok:
                        ok,why=CommitGate.admit(candidate,t,pipeline['checked_version'],map_version,pipeline['expected'],state_at(candidate['start']))
                    if ok and latest['stamp']>map_stamp+1e-9:ok=False;why='newer_scan_waiting'
                    if time.perf_counter()-begin>CONFIG['dt']:ok=False;why='commit_cpu_budget'
                    if ok:
                        pending=Curve(candidate);accepted+=1;plans.append(candidate)
                        events.append(dict(kind='commit',t=t,start=candidate['start'],cycle_s=t-pipeline['started'],map_stamp=map_stamp,map_version=map_version,route_source=pipeline['route_source']))
                    else:record_failure(why);events.append(dict(kind='commit_rejected',t=t,reason=why))
                    pipeline=None;next_plan=t+.1
            if pipeline is None and pending is None and t>=next_plan-1e-9:
                expected=state_at(t+CONFIG['handoff'])
                request=cloud_if_new(dict(op='plan',t=t,start_state=expected,goal=scene['goal']))
                result=req(w,request);map_stamp=result['map_stamp'];map_version=result['map_version']
                pipeline=dict(phase='solve',ready=round_tick(t+result['rpc_ms']/1000+extra_delay),result=result,expected=expected,started=t)
                next_plan=t+.4
            t=round(t+CONFIG['dt'],10)
    traj=np.asarray(trajectory);np.savetxt(out/'trajectory.csv',traj,delimiter=',',header='t,x,y,z,vx,vy,vz,ax,ay,az,clearance_minus_planning_radius',comments='')
    save(out/'events.json',events);save(out/'plans.json',plans);save(out/'timings.json',rpc)
    def pct(values):return {f'p{q}':float(np.percentile(values,q)) for q in [50,95,99]} if values else {}
    # Signed distance is 1-Lipschitz. Derivative-control bounds limit speed to 3 m/s.
    # Subtract v_max*dt/2 for the nearest sample in each continuous interval.
    lower_clearance=min_gap-CONFIG['max_velocity']*CONFIG['dt']/2
    routing=[x for x in rpc if x['op']=='plan']
    result=dict(backend=backend,scene=scene['name'],seed=scene['seed'],extra_delay_s=extra_delay,outcome=reason,route_policy=effective_config['route_policy'],
        success=reason=='reached',duration_s=t-1,final_error_m=float(np.linalg.norm(traj[-1,1:4]-scene['goal'])),
        path_length_m=float(np.linalg.norm(np.diff(traj[:,1:4],axis=0),axis=1).sum()),min_sample_clearance_m=min_gap,
        conservative_interval_clearance_m=lower_clearance,peak_velocity=peak_v,peak_acceleration=peak_a,
        accepted_plans=accepted,rejections=failures,rpc_ms=pct([x['rpc_ms'] for x in rpc[4:]]),
        update_ms=pct([x['map_update_ms'] for x in rpc[4:] if x['map_update_ms']>0]),
        commit_cycle_ms=pct([x['cycle_s']*1000 for x in events if x['kind']=='commit']),
        sensor_render_ms=pct(generation),wall_seconds=time.perf_counter()-start_wall,
        config_sha256=digest(effective_config),scene_sha256=digest(scene),
        full_astar_calls=sum(x.get('full_astar_calls',0) for x in routing),local_astar_calls=sum(x.get('local_astar_calls',0) for x in routing),
        reused_routes=sum(x.get('route_source')=='reuse' for x in routing),repaired_routes=sum(x.get('route_source')=='repair' for x in routing),
        search_ms=pct([x['search_ms'] for x in routing if 'search_ms' in x]),astar_ms=pct([x['astar_ms'] for x in routing if 'astar_ms' in x]))
    save(out/'result.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['ego','rog','both'],default='both');p.add_argument('--scenes',nargs='+',default=['empty','wall_gap','offset_wall','forest']);p.add_argument('--seeds',nargs='+',type=int,default=[41,42,43]);p.add_argument('--delays',nargs='+',type=float,default=[0.,.08]);p.add_argument('--max-time',type=float,default=35.);p.add_argument('--out',type=Path,required=True);p.add_argument('--route-policy',choices=['full','reuse','repair'],default=None);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);results=[]
    for name in args.scenes:
        for seed in (args.seeds if name=='forest' else [0]):
            for delay in args.delays:
                for backend in (['ego','rog'] if args.backend=='both' else [args.backend]):
                    out=args.out/f'{name}_{seed}_{backend}_delay{round(delay*1000)}'
                    if (out/'result.json').exists():raise FileExistsError(f'Refusing to overwrite experiment {out}')
                    r=trial(backend,make_scene(name,seed),delay,out,args.max_time,args.route_policy);results.append(r);save(args.out/'summary.json',results)
                    print(json.dumps({k:r[k] for k in ['backend','scene','seed','extra_delay_s','outcome','duration_s','final_error_m','accepted_plans','rejections']}),flush=True)
if __name__=='__main__':main()
