"""Controlled sensor impairments around the unchanged navigation worker/executor."""
import heapq,time
from collections import Counter
from navigation_common import *

def conditions():
    base=dict(rate_hz=5.,azimuths=720,delay_s=0.,drop_probability=0.,burst_period_s=0.,burst_length_s=0.)
    def case(name,family,**kwargs):return dict(base,name=name,family=family,**kwargs)
    result=[case('baseline','baseline')]
    result += [case(f'rate_frame_{hz:g}','rate_fixed_frame',rate_hz=hz) for hz in [2.5,10.,20.]]
    result += [case(f'rate_budget_{hz:g}','rate_fixed_points_per_second',rate_hz=hz,azimuths=int(720*5/hz)) for hz in [2.5,10.,20.]]
    result += [case(f'points_{az*17}','point_count',azimuths=az) for az in [90,180,360]]
    result += [case(f'delay_{ms}','sensor_transport_delay',delay_s=ms/1000) for ms in [50,100,200,400]]
    result += [case(f'drop_{pct}','independent_drop',drop_probability=pct/100) for pct in [10,30,50]]
    result += [case('burst_600ms','burst_drop',burst_period_s=2.,burst_length_s=.6)]
    return result

class Channel:
    def __init__(self,settings,seed):
        self.settings=settings;self.rng=np.random.default_rng(seed)
        self.sensor=Sensor(settings['azimuths']);self.queue=[];self.records=[];self.latest=None;self.next_seq=0
    def due(self,t):return t>=1.+self.next_seq/self.settings['rate_hz']-1e-9
    def capture(self,scene,p,t):
        sequence=self.next_seq;expected=1.+sequence/self.settings['rate_hz']
        if abs(expected-t)>CONFIG['dt']+1e-8:raise ValueError('missed capture event')
        self.next_seq+=1;u=float(self.rng.random());s=self.settings
        burst=s['burst_period_s']>0 and ((expected-1.)%s['burst_period_s'])<s['burst_length_s']-1e-8
        dropped=u<s['drop_probability'] or burst
        record=dict(sequence=sequence,stamp=t,origin=list(p),scheduled_arrival=t+s['delay_s'],dropped=bool(dropped),uniform=u,
                    point_count=len(self.sensor.d),delivered_at=None,mapped_at=None)
        self.records.append(record)
        # A discarded frame need not be ray-rendered; the sensor work is not planner work.
        if not dropped:
            cloud=self.sensor.capture(scene,p,t);cloud['sequence']=sequence
            if cloud.get('sensor')=='d435i':
                # Ray rendering is simulation-only; actual depth projection cost
                # is charged to arrival latency, before planning can see the frame.
                record['projection_ms']=cloud['projection_ms']
                record['scheduled_arrival']+=cloud['projection_ms']/1000
            heapq.heappush(self.queue,(record['scheduled_arrival'],sequence,cloud))
    def deliver(self,t):
        while self.queue and self.queue[0][0]<=t+1e-9:
            _,sequence,cloud=heapq.heappop(self.queue)
            self.records[sequence]['delivered_at']=t
            if self.latest is None or cloud['stamp']>self.latest['stamp']:
                self.latest=cloud
        return self.latest
    def used(self,cloud,t):
        record=self.records[cloud['sequence']]
        assert not record['dropped'] and record['delivered_at'] is not None
        assert record['stamp']<=record['scheduled_arrival']<=t+1e-8
        assert record['mapped_at'] is None,'same scan consumed twice'
        record['mapped_at']=t

def percentiles(values):
    return {f'p{q}':float(np.percentile(values,q)) for q in [50,95,99]} if values else {}

def trial(backend,scene,settings,channel_seed,out,max_time=35.,extra_delay=.08):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    effective=dict(CONFIG,route_policy='repair')
    save(out/'config.json',dict(effective,backend=backend,extra_delay_s=extra_delay,sensor_experiment=settings,channel_seed=channel_seed))
    save(out/'scene.json',scene)
    channel=Channel(settings,channel_seed);events=[];trajectory=[];rpc=[];generation=[];plans=[];wait_for_scan_rejections=0
    initial=stationary(scene['start']);active=None;pending=None;pipeline=None
    t=1.;next_plan=t;latest=None;map_stamp=.6;map_version=4
    reason='timeout';failures=Counter();accepted=0;peak_v=0.;peak_a=0.;min_gap=100.;start_wall=time.perf_counter()
    stopped_s=0.;stop_episodes=0;was_stopped=True;has_moved=False;busy_s=0.
    def state_at(at_t):return active.state(at_t) if active else initial
    def req(w,request):
        if 'cloud' in request and 'sequence' in request['cloud']:channel.used(request['cloud'],request['t'])
        r=w.request(request)
        row=dict(t=request['t'],op=request['op'],**{k:v for k,v in r.items() if k not in ['plan','route','cells']})
        row['map_age_ms']=1000*(request['t']-r['map_stamp'])
        row['included_scan']='cloud' in request
        if 'cloud' in request:row['input_stamp']=request['cloud']['stamp']
        rpc.append(row);return r
    def cloud_if_new(request):
        if latest is not None and latest['stamp']>map_stamp:request['cloud']=latest
        if active is not None:request['executing_plan']=active.plan
        return request
    with Worker(backend,out/'worker',route_policy='repair') as w:
        # Common start map for every condition. Stress starts after this stationary boot.
        boot_sensor=Sensor()
        for i in range(4):req(w,dict(op='update',t=i*.2,cloud=boot_sensor.capture(scene,scene['start'],i*.2)))
        while t<=1.+max_time+1e-9:
            if pending is not None and t>=pending.start-1e-9:
                before=state_at(pending.start);after=pending.state(pending.start)
                jump={k:float(np.linalg.norm(np.asarray(before[k])-after[k])) for k in ['p','v','a']}
                events.append(dict(kind='activate',t=t,jump=jump))
                if jump['p']>1e-6 or jump['v']>1e-6 or jump['a']>1e-5:reason='discontinuous_handoff';break
                active=pending;pending=None
            current=state_at(t);p=np.asarray(current['p']);v=np.asarray(current['v']);a=np.asarray(current['a'])
            gap=float(clearance(scene,p)-RADIUS);min_gap=min(min_gap,gap);peak_v=max(peak_v,float(np.linalg.norm(v)));peak_a=max(peak_a,float(np.linalg.norm(a)))
            trajectory.append([t,*p,*v,*a,gap,map_stamp])
            if gap<0:reason='envelope_collision';break
            if not CONFIG['height_band'][0]-1e-8<=p[2]<=CONFIG['height_band'][1]+1e-8:reason='height_violation';break
            if np.linalg.norm(p-scene['goal'])<.18 and np.linalg.norm(v)<.15:reason='reached';break
            moving=np.linalg.norm(v)>=.1
            if moving:has_moved=True
            if has_moved and not moving:
                stopped_s+=CONFIG['dt']
                if not was_stopped:stop_episodes+=1
            was_stopped=not moving
            if pipeline is not None:busy_s+=CONFIG['dt']
            if channel.due(t):
                begin=time.perf_counter();channel.capture(scene,p,t);generation.append((time.perf_counter()-begin)*1000)
            latest=channel.deliver(t)
            if pipeline is not None and t>=pipeline['ready']-1e-9:
                phase=pipeline['phase'];r=pipeline['result']
                if phase=='solve':
                    if r.get('reason')=='unsafe_execution_prefix':reason='unsafe_execution_prefix';break
                    if not r['ok']:
                        failures[r['reason']]+=1;events.append(dict(kind='solve_rejected',t=t,reason=r['reason']));pipeline=None;next_plan=t+.1
                    else:
                        request=cloud_if_new(dict(op='validate',t=t,plan=r['plan'],start_state=pipeline['expected']))
                        validated=req(w,request);map_stamp=validated['map_stamp'];map_version=validated['map_version']
                        pipeline=dict(phase='validate',ready=round_tick(t+validated['rpc_ms']/1000+CONFIG['publication_delay']+CONFIG['dt']),result=validated,plan=r['plan'],expected=pipeline['expected'],checked_version=map_version,started=pipeline['started'],route_source=r.get('route_source','full'),solve_stamp=pipeline['solve_stamp'])
                else:
                    begin=time.perf_counter();candidate=pipeline['plan'];why=r.get('reason','invalid');ok=r['ok']
                    if why=='unsafe_execution_prefix':reason='unsafe_execution_prefix';break
                    if ok:ok,why=CommitGate.admit(candidate,t,pipeline['checked_version'],map_version,pipeline['expected'],state_at(candidate['start']))
                    if ok and latest is not None and latest['stamp']>map_stamp+1e-9:ok=False;why='newer_scan_waiting'
                    if time.perf_counter()-begin>CONFIG['dt']:ok=False;why='commit_cpu_budget'
                    if ok:
                        pending=Curve(candidate);accepted+=1;plans.append(candidate)
                        events.append(dict(kind='commit',t=t,start=candidate['start'],cycle_s=t-pipeline['started'],map_stamp=map_stamp,map_version=map_version,route_source=pipeline['route_source'],map_age_ms=(t-map_stamp)*1000,solve_cloud_to_activation_ms=(candidate['start']-pipeline['solve_stamp'])*1000,validated_cloud_to_activation_ms=(candidate['start']-map_stamp)*1000))
                    else:
                        failures[why]+=1;events.append(dict(kind='commit_rejected',t=t,reason=why,map_stamp=map_stamp,latest_stamp=latest['stamp'] if latest else None))
                    pipeline=None;next_plan=t+.1
            if pipeline is None and pending is None and t>=next_plan-1e-9:
                expected=state_at(t+CONFIG['handoff']);request=cloud_if_new(dict(op='plan',t=t,start_state=expected,goal=scene['goal']))
                result=req(w,request);map_stamp=result['map_stamp'];map_version=result['map_version']
                pipeline=dict(phase='solve',ready=round_tick(t+result['rpc_ms']/1000+extra_delay),result=result,expected=expected,started=t,solve_stamp=map_stamp)
                next_plan=t+.4
            t=round(t+CONFIG['dt'],10)
    traj=np.asarray(trajectory)
    np.savetxt(out/'trajectory.csv',traj,delimiter=',',header='t,x,y,z,vx,vy,vz,ax,ay,az,clearance_minus_planning_radius,map_stamp',comments='')
    for filename,data in [('events.json',events),('plans.json',plans),('timings.json',rpc),('sensor_frames.json',channel.records)]:save(out/filename,data)
    commits=[x for x in events if x['kind']=='commit'];routing=[x for x in rpc if x['op']=='plan'];allrpc=rpc[4:]
    mapped=[x for x in channel.records if x['mapped_at'] is not None];delivered=[x for x in channel.records if x['delivered_at'] is not None]
    duration=t-1;distance=float(np.linalg.norm(np.diff(traj[:,1:4],axis=0),axis=1).sum())
    result=dict(backend=backend,scene=scene['name'],seed=scene['seed'],condition=settings['name'],family=settings['family'],settings=settings,channel_seed=channel_seed,
        outcome=reason,success=reason=='reached',duration_s=duration,final_error_m=float(np.linalg.norm(traj[-1,1:4]-scene['goal'])),path_length_m=distance,
        min_sample_clearance_m=min_gap,conservative_interval_clearance_m=min_gap-CONFIG['max_velocity']*CONFIG['dt']/2,
        peak_velocity=peak_v,peak_acceleration=peak_a,accepted_plans=accepted,rejections=dict(failures),
        stopped_after_motion_s=stopped_s,stop_episodes=stop_episodes,pipeline_busy_s=busy_s,
        frames_captured=len(channel.records),frames_dropped=sum(x['dropped'] for x in channel.records),frames_delivered=len(delivered),frames_mapped=len(mapped),
        delivered_unmapped_frames=len(delivered)-len(mapped),nominal_points_per_frame=len(channel.sensor.d),nominal_points_per_second=len(channel.sensor.d)*settings['rate_hz'],
        map_input_age_ms=percentiles([(x['mapped_at']-x['stamp'])*1000 for x in mapped]),
        planning_map_age_ms=percentiles([x['map_age_ms'] for x in routing]),
        mapping_rate_hz=len(mapped)/max(duration,1e-9),rpc_ms=percentiles([x['rpc_ms'] for x in allrpc]),
        plan_rpc_ms=percentiles([x['rpc_ms'] for x in routing]),update_ms=percentiles([x['map_update_ms'] for x in allrpc if x['map_update_ms']>0]),
        search_ms=percentiles([x['search_ms'] for x in routing if 'search_ms' in x]),
        opt_validate_ms=percentiles([x['opt_validate_ms'] for x in routing if 'opt_validate_ms' in x]),
        commit_cycle_ms=percentiles([x['cycle_s']*1000 for x in commits]),
        solve_cloud_to_activation_ms=percentiles([x['solve_cloud_to_activation_ms'] for x in commits]),
        validated_cloud_to_activation_ms=percentiles([x['validated_cloud_to_activation_ms'] for x in commits]),
        sensor_render_ms=percentiles(generation),wall_seconds=time.perf_counter()-start_wall,config_sha256=digest(effective),scene_sha256=digest(scene),
        total_rpc_cpu_ms=sum(x['rpc_ms'] for x in allrpc),map_cpu_ms=sum(x['map_update_ms'] for x in allrpc),
        search_cpu_ms=sum(x.get('search_ms',0) for x in routing),optimization_check_cpu_ms=sum(x.get('opt_validate_ms',0) for x in routing),
        full_astar_calls=sum(x.get('full_astar_calls',0) for x in routing),local_astar_calls=sum(x.get('local_astar_calls',0) for x in routing))
    save(out/'result.json',result);return result
