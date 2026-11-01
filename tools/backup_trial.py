"""P0: committed nominal/stop schedules, serialized sensing/compute, ideal execution.

The 0.5 s gate and latest-arrived-frame commit gate are unchanged. A stop is
committed with the nominal curve; activating it does not create a new command.
"""
from perception_trial import Channel,conditions,percentiles
from navigation_common import *
from collections import Counter

def hold_plan(p):
    return dict(start=0.,dt=1.,duration=3.,controls=[list(p)]*6,kind='hold')

def trial(backend,scene,settings,channel_seed,out,max_time=35.,extra_delay=.08,timeout_after=None,obstacle_event=None,replay_prefix=None,frontend=None,resume_from=None,plan_budget_ms=None,camera_yaw=None,recovery_mode=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    save(out/'config.json',dict(CONFIG,backend=backend,sensor_experiment=settings,channel_seed=channel_seed,
        extra_delay=extra_delay,timeout_after=timeout_after,obstacle_event=obstacle_event,backup_branch_offset=.6,recovery_mode=recovery_mode,replay_prefix=str(replay_prefix) if replay_prefix else None))
    save(out/'scene.json',scene)
    if frontend is not None:save(out/'model.json',frontend.metadata)
    channel=Channel(settings,channel_seed);schedule=[Curve(hold_plan(scene['start']))]
    pipeline=None;latest=None;map_stamp=.6;map_version=4;next_plan=1.;t=1.
    events=[];traffic=[];plans=[];trajectory=[];rejected=Counter();reason='timeout';changed_scene=False
    accepted=0;stops=0;start_wall=time.perf_counter();guarantee_lost=False
    def state_at(at):
        selected=schedule[0]
        for c in schedule[1:]:
            if c.start<=at+1e-9:selected=c
        return selected.state(at)
    def request(w,q):
        q=dict(q,execution_schedule=[c.plan for c in schedule])
        nominal=next((c for c in schedule if c.plan['kind']=='nominal'),None)
        if nominal:q['nominal_plan']=nominal.plan
        if latest is not None and latest['stamp']>map_stamp+1e-9:
            q['cloud']=latest;channel.used(latest,t)
        r=w.request(q)
        traffic.append(dict(t=t,op=q['op'],included_scan='cloud' in q,**{k:v for k,v in r.items() if k not in ['plan','backup_plan']}))
        return r
    with Worker(backend,out/'worker',route_policy='repair',frontend=frontend) as w:
        if plan_budget_ms is not None:w.plan_budget_ms=plan_budget_ms
        sensor=Sensor()
        if resume_from:
            from branch_resume import restore
            t,latest,map_stamp,map_version,schedule,proof=restore(w,frontend,scene,channel,*resume_from)
            next_plan=t;save(out/'branch_initialization.json',proof)
        elif replay_prefix:
            folder=Path(replay_prefix)
            original=[json.loads(line) for line in (folder/'worker/requests.jsonl').read_text().splitlines()]
            for record in original:
                cloud=record['request'].get('cloud')
                if cloud and 'binary_xyz_f32' in cloud:
                    cloud['binary_xyz_f32']=str((folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name).resolve())
            failed=next(x for x in original if x['response'].get('reason')=='unsafe_execution_prefix')
            target=failed['request']['executing_plan']
            validation=next(i for i,x in enumerate(original) if x['request']['op']=='validate' and x['response']['ok'] and digest(x['request']['plan'])==digest(target))
            for x in original[:validation]:
                q=x['request']
                if 'cloud' in q:w.request(dict(op='update',t=q['t'],cloud=q['cloud']))
            q=dict(original[validation]['request']);previous=dict(q.pop('executing_plan'),kind='nominal')
            q.update(execution_schedule=[previous],prepare_backup=True)
            prepared=w.request(q)
            ready=round_tick(q['t']+prepared['rpc_ms']/1000+CONFIG['publication_delay']+CONFIG['dt'])
            assert prepared['ok'] and ready<target['start']-CONFIG['deadline_guard'], 'prefix branch preparation missed original deadline'
            t=ready;map_stamp=prepared['map_stamp'];map_version=prepared['map_version']
            # Replay only captures before divergence. After this point all new
            # observations are ray-rendered from the new actually executed path.
            for frame in json.loads((folder/'sensor_frames.json').read_text()):
                if frame['stamp']>t+1e-9:break
                channel.capture(scene,frame['origin'],frame['stamp'])
                if frame['mapped_at'] is not None and frame['mapped_at']<=q['t']+1e-9:channel.records[-1]['mapped_at']=frame['mapped_at']
            latest=channel.deliver(t)
            for frame in json.loads((folder/'sensor_frames.json').read_text()):
                if frame['stamp']>t+1e-9:break
                if frame['delivered_at'] is not None and frame['delivered_at']<=t+1e-9:
                    channel.records[frame['sequence']]['delivered_at']=frame['delivered_at']
            assert latest is None or latest['stamp']<=map_stamp+1e-9,'newer scan at prefix commit'
            nominal=dict(target,kind='nominal');backup=dict(prepared['backup_plan'],kind='backup')
            schedule=[Curve(previous),Curve(nominal),Curve(backup)];plans.append(dict(nominal=nominal,backup=backup));accepted=1
            next_plan=t+.1
            proof=dict(kind='commit',t=t,plan_index=0,start=target['start'],backup_start=backup['start'],map_stamp=map_stamp,map_version=map_version,
                       prefix_source=str(folder),original_validate_time=q['t'],preparation_rpc_ms=prepared['rpc_ms'])
            events.append(proof);save(out/'prefix_initialization.json',proof)
        else:
            for i in range(4):w.request(dict(op='update',t=i*.2,cloud=sensor.capture(scene,scene['start'],i*.2)))
        start_t=t
        while t<=start_t+max_time+1e-9:
            while len(schedule)>1 and schedule[1].start<=t+1e-9:
                old=schedule.pop(0);new=schedule[0];before=old.state(new.start);after=new.state(new.start)
                jumps={k:float(np.linalg.norm(np.asarray(before[k])-after[k])) for k in ['p','v','a']}
                events.append(dict(kind='activate',curve_kind=new.plan['kind'],t=t,start=new.start,jump=jumps,pipeline_busy=pipeline is not None))
                if any(jumps[k]>tol for k,tol in [('p',1e-6),('v',1e-6),('a',1e-5)]):
                    reason='discontinuous_handoff';guarantee_lost=True;break
                if new.plan['kind']=='backup':stops+=1
            if guarantee_lost:break
            current=state_at(t);p=np.asarray(current['p']);v=np.asarray(current['v']);a=np.asarray(current['a'])
            if camera_yaw is not None:camera_yaw.update(t,current,scene['goal'])
            if obstacle_event and not changed_scene and t>=obstacle_event['t']-1e-9:
                scene=dict(scene,obstacles=scene['obstacles']+[obstacle_event['obstacle']]);changed_scene=True
                events.append(dict(kind='obstacle_appeared',t=t,obstacle=obstacle_event['obstacle']))
            gap=float(clearance(scene,p)-RADIUS);trajectory.append([t,*p,*v,*a,gap,map_stamp])
            if gap<0:reason='envelope_collision';break
            if np.linalg.norm(p-scene['goal'])<.18 and np.linalg.norm(v)<.15:reason='reached';break
            if channel.due(t):
                channel.capture(scene,p,t)
                for _,sequence,cloud in channel.queue:
                    if sequence==channel.next_seq-1:cloud['capture_state']=current
            latest=channel.deliver(t)
            if pipeline and t>=pipeline['ready']-1e-9:
                item=pipeline;r=item['result'];pipeline=None
                checked=r.get('execution_guard',r if item['phase']=='guard' else None)
                if checked:
                    events.append(dict(kind='execution_check',requested_at=item['requested_at'],t=t,map_version=r['map_version'],report=checked))
                    if not checked['ok']:
                        if not recovery_mode:
                            reason='no_validated_escape';guarantee_lost=True;break
                        escape=request(w,dict(op='escape',t=t));map_stamp=escape['map_stamp'];map_version=escape['map_version']
                        pipeline=dict(phase='escape',ready=round_tick(t+escape['rpc_ms']/1000+CONFIG['publication_delay']+CONFIG['dt']),result=escape,requested_at=t,checked_version=map_version)
                        events.append(dict(kind='escape_search',t=t,ready=pipeline['ready'],report=escape))
                        t=round(t+CONFIG['dt'],10);continue
                if item['phase']=='escape':
                    # New observations cannot be ignored while publishing a stop.
                    if latest is not None and latest['stamp']>map_stamp+1e-9:
                        escape=request(w,dict(op='escape',t=t));map_stamp=escape['map_stamp'];map_version=escape['map_version']
                        pipeline=dict(phase='escape',ready=round_tick(t+escape['rpc_ms']/1000+CONFIG['publication_delay']+CONFIG['dt']),result=escape,requested_at=t,checked_version=map_version)
                        events.append(dict(kind='escape_refresh',t=t,ready=pipeline['ready'],report=escape))
                    else:
                        selected=None;checks=[]
                        for entry in r['stops']:
                            plan=entry['plan'];expected=Curve(plan).state(plan['start'])
                            ok,why=CommitGate.admit(plan,t,item['checked_version'],map_version,expected,state_at(plan['start']))
                            if t-map_stamp>.5:ok=False;why='stale_cloud'
                            first=r['prior_guard']['first_invalid_time']
                            if first is not None and t>=first:ok=False;why='invalid_prefix_before_commit'
                            checks.append(dict(start=plan['start'],ok=ok,reason=why))
                            if ok:selected=dict(plan,kind='backup');break
                        if selected is None:
                            events.append(dict(kind='escape_failed',t=t,checks=checks,report=r));reason='no_validated_escape';guarantee_lost=True;break
                        schedule=[c for c in schedule if c.start<selected['start']-1e-9]+[Curve(selected)]
                        events.append(dict(kind='escape_commit',t=t,start=selected['start'],stop=selected,requested_at=item['requested_at'],map_stamp=map_stamp,map_version=map_version,checks=checks))
                        next_plan=t+.1
                elif item['phase']=='solve':
                    if not r['ok']:
                        rejected[r['reason']]+=1;events.append(dict(kind='solve_rejected',t=t,reason=r['reason']));next_plan=t+.1
                    else:
                        checked=request(w,dict(op='validate',t=t,plan=r['plan'],backup_plan=r['backup_plan'],start_state=item['expected']))
                        map_stamp=checked['map_stamp'];map_version=checked['map_version']
                        pipeline=dict(phase='validate',ready=round_tick(t+checked['rpc_ms']/1000+CONFIG['publication_delay']+CONFIG['dt']),
                            result=checked,plan=r['plan'],backup=r['backup_plan'],expected=item['expected'],checked_version=map_version,requested_at=t)
                elif item['phase']=='validate':
                    candidate=item['plan'];ok=r['ok'];why=r.get('reason','invalid')
                    begin=time.perf_counter()
                    if ok:ok,why=CommitGate.admit(candidate,t,item['checked_version'],map_version,item['expected'],state_at(candidate['start']))
                    if ok and latest is not None and latest['stamp']>map_stamp+1e-9:ok=False;why='newer_scan_waiting'
                    if time.perf_counter()-begin>CONFIG['dt']:ok=False;why='commit_cpu_budget'
                    if ok:
                        candidate=dict(candidate,kind='nominal');backup=dict(item['backup'],kind='backup')
                        schedule=[c for c in schedule if c.start<candidate['start']-1e-9]+[Curve(candidate),Curve(backup)]
                        plans.append(dict(nominal=candidate,backup=backup));accepted+=1
                        events.append(dict(kind='commit',t=t,plan_index=len(plans)-1,start=candidate['start'],backup_start=backup['start'],
                            backup_commit_deadline=backup['start']-CONFIG['deadline_guard'],map_stamp=map_stamp,map_version=map_version))
                    else:
                        rejected[why]+=1;events.append(dict(kind='commit_rejected',t=t,reason=why))
                    next_plan=t+.1
            if pipeline is None:
                pending=any(c.plan['kind']=='nominal' and c.start>t+1e-9 for c in schedule)
                if not pending and t>=next_plan-1e-9:
                    expected=state_at(t+CONFIG['handoff'])
                    r=request(w,dict(op='plan',t=t,start_state=expected,goal=scene['goal'],with_backup=True,execution_delay_s=extra_delay))
                    map_stamp=r['map_stamp'];map_version=r['map_version']
                    delay=.65 if timeout_after is not None and t>=timeout_after else extra_delay
                    if recovery_mode and not r.get('execution_guard',dict(ok=True))['ok']:delay=0.
                    pipeline=dict(phase='solve',ready=round_tick(t+r['rpc_ms']/1000+delay),result=r,expected=expected,requested_at=t)
                elif latest is not None and latest['stamp']>map_stamp+1e-9:
                    r=request(w,dict(op='guard',t=t));map_stamp=r['map_stamp'];map_version=r['map_version']
                    pipeline=dict(phase='guard',ready=round_tick(t+r['rpc_ms']/1000),result=r,requested_at=t)
            t=round(t+CONFIG['dt'],10)
    traj=np.asarray(trajectory)
    if camera_yaw is not None:save(out/'camera_yaw.json',camera_yaw.records)
    if resume_from and reason=='timeout':reason='branch_horizon'
    np.savetxt(out/'trajectory.csv',traj,delimiter=',',header='t,x,y,z,vx,vy,vz,ax,ay,az,clearance_minus_planning_radius,map_stamp',comments='')
    for name,data in [('plans',plans),('events',events),('timings',traffic),('sensor_frames',channel.records)]:save(out/(name+'.json'),data)
    result=dict(backend=backend,scene=scene['name'],condition=settings['name'],outcome=reason,success=reason=='reached',
        duration_s=t-start_t,initial_time=start_t,final_time=t,final_error_m=float(np.linalg.norm(traj[-1,1:4]-scene['goal'])),final_speed=float(np.linalg.norm(traj[-1,4:7])),
        accepted_plans=accepted,backup_activations=stops,rejections=dict(rejected),guarantee_lost=guarantee_lost,
        min_sample_clearance_m=float(traj[:,10].min()),peak_velocity=float(np.linalg.norm(traj[:,4:7],axis=1).max()),
        peak_acceleration=float(np.linalg.norm(traj[:,7:10],axis=1).max()),rpc_ms=percentiles([r['rpc_ms'] for r in traffic]),
        wall_seconds=time.perf_counter()-start_wall,final_schedule=[c.plan for c in schedule])
    save(out/'result.json',result);return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['ego','rog'],required=True);p.add_argument('--condition',default='delay_400')
    p.add_argument('--scene',default='offset_wall');p.add_argument('--seed',type=int,default=0);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-time',type=float,default=35.);p.add_argument('--timeout-after',type=float);p.add_argument('--replay-prefix',type=Path);a=p.parse_args()
    r=trial(a.backend,make_scene(a.scene,a.seed),next(c for c in conditions() if c['name']==a.condition),12000,a.out,a.max_time,timeout_after=a.timeout_after,replay_prefix=a.replay_prefix)
    print(json.dumps(r),flush=True)
