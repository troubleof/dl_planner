"""Privileged future RGB-D renderings on the candidate's OWN executed branch.

Only label generation uses truth geometry. Starts from the original received
map prefix and pose/yaw, keeps original pending pre-branch frame arrivals, and
ray-renders all later frames at candidate/stop positions with bounded yaw.
"""
import argparse,copy,hashlib,json,time,heapq
from pathlib import Path
import numpy as np
from navigation_common import Worker,Curve,save,CONFIG,round_tick
from camera_yaw import CameraYaw,install
from recovery_frontend import HORIZONS

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--states',type=int,default=64);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    loaded=np.load(a.dataset/'dataset.npz');original={k:loaded[k] for k in loaded.files};arrays={k:[x.copy() for x in original[k]] for k in original};rows=json.loads((a.dataset/'records.json').read_text());new_rows=copy.deepcopy(rows);rng=np.random.default_rng(202610317);details=[];begin=time.perf_counter()
    priority=[]
    for i,row in enumerate(rows):
        current=original['labels'][i,:,0]>.5
        if not current.any():continue
        # Deliberately include near-boundary, moving, failed-episode and rare
        # future-invalidation states; selection uses only training/validation.
        features=original['features'][i];unknown=features[current,6::3][:,:5].mean();speed=np.linalg.norm(row['query']['start_state']['v']);negative=((original['labels'][i,:,1:]<.5)&(original['mask'][i,:,1:]>0)&current[:,None]).sum()
        priority.append((100*bool(negative)+5*(row['outcome']!='reached')+unknown+speed/10+rng.uniform(0,.1),i))
    selected=[]
    for split,count in [('train',int(a.states*.75)),('validation',a.states-int(a.states*.75))]:
        eligible=sorted([x for x in priority if rows[x[1]]['split']==split],reverse=True);selected.extend(i for _,i in eligible[:count])
    for state_id in selected:
        row=rows[state_id];case=Path(row['source_case']);scene=json.loads((case/'scene.json').read_text());traj=np.loadtxt(case/'trajectory.csv',delimiter=',',skiprows=1);yaw_log=np.asarray(json.loads((case/'camera_yaw.json').read_text()));frames=json.loads((case/'sensor_frames.json').read_text());t0=row['query']['t'];valid=np.flatnonzero(original['labels'][state_id,:,0]>.5);score=original['features'][state_id,:,-1]
        negative=[j for j in valid if np.any((original['labels'][state_id,j,1:]<.5)&(original['mask'][state_id,j,1:]>0))];ordered=sorted(valid,key=lambda j:-score[j]);choices=list(dict.fromkeys(negative[:1]+ordered[:1]+ordered[len(ordered)//2:len(ordered)//2+1]+ordered[-1:]))[:4]
        for j in choices:
            schedule=next(s['execution_schedule'] for s in row['schedules'] if s['id']==j);curves=[Curve(p) for p in schedule]
            def state_at(t):
                active=curves[0]
                for c in curves[1:]:
                    if c.start<=t+1e-9:active=c
                return active.state(t)
            yaw=CameraYaw(row['policy'] not in ['repair','old','short_fixed'])
            for tr in traj[traj[:,0]<=t0+1e-9]:yaw.update(tr[0],dict(p=tr[1:4],v=tr[4:7],a=tr[7:10]),scene['goal'])
            original_yaw=yaw_log[np.argmin(np.abs(yaw_log[:,0]-t0))]
            assert abs(yaw.yaw-original_yaw[1])<1e-7 and abs(yaw.rate-original_yaw[2])<1e-7,(state_id,row['policy'],yaw.yaw,yaw.rate,original_yaw.tolist())
            install(yaw)
            import perception_trial
            sensor=perception_trial.Sensor();queue=[];seq=0;captures=[];proof=[];alive=True;label=np.zeros_like(original['labels'][state_id]);mask=np.zeros_like(label);label[j,0]=1;mask[j,0]=1;done=[False]*3
            # A captured but not yet delivered frame remains in flight; it is
            # not given to the model input or silently dropped from the branch.
            for f in frames:
                if f['stamp']<=t0+1e-9 and f['scheduled_arrival']>t0+1e-9 and not f['dropped']:
                    at=yaw_log[np.argmin(abs(yaw_log[:,0]-f['stamp']))];saved=yaw.yaw;yaw.yaw=float(at[1]);cloud=sensor.capture(scene,f['origin'],f['stamp']);yaw.yaw=saved
                    heapq.heappush(queue,(f['scheduled_arrival'],seq,cloud));seq+=1
            next_capture=1.+(np.floor((t0-1.+1e-8)/.2)+1)*.2;name=f'state{state_id:05d}_candidate{j:02d}'
            with Worker(row['backend'],a.out/'workers'/name) as w:
                for ev in row['prefix']:w.request(dict(op='update',t=ev['t'],cloud=ev['cloud']))
                mapped_stamp=row['prefix'][-1]['cloud']['stamp']
                for step in range(1,int(round(max(HORIZONS)/CONFIG['dt']))+1):
                    t=round(t0+step*CONFIG['dt'],10);s=state_at(t);yaw.update(t,s,scene['goal'])
                    if t>=next_capture-1e-9:
                        cloud=sensor.capture(scene,s['p'],t);cloud['capture_state']=s;arrival=t+cloud['projection_ms']/1000;heapq.heappush(queue,(arrival,seq,cloud));seq+=1;next_capture+=.2;captures.append(dict(stamp=t,arrival=arrival,p=s['p'],yaw=yaw.yaw))
                    newest=None
                    while queue and queue[0][0]<=t+1e-9:
                        arrival,_,cloud=heapq.heappop(queue)
                        if cloud['stamp']>mapped_stamp and (newest is None or cloud['stamp']>newest['stamp']):newest=cloud
                    if newest is not None:
                        cloud=newest;w.request(dict(op='update',t=t,cloud=cloud));mapped_stamp=cloud['stamp']
                        checked=w.request(dict(op='label_schedules',t=t,schedules=[dict(id=int(j),execution_schedule=schedule)]))['branches'][0];alive=alive and checked['ok'];proof.append(dict(t=t,stamp=cloud['stamp'],report=checked))
                    for h,dt in enumerate(HORIZONS):
                        if not done[h] and t>=t0+dt-1e-9:label[j,h+1]=alive;mask[j,h+1]=1;done[h]=True
                assert all(done)
            idx=len(new_rows);newrow=dict(row,index=idx,parent_state=state_id,label_source='own_candidate_future_camera',label_candidate=int(j));new_rows.append(newrow)
            for key in arrays:arrays[key].append(label if key=='labels' else mask if key=='mask' else original[key][state_id].copy())
            item=dict(state=idx,parent_state=state_id,candidate=int(j),split=row['split'],labels=label[j].tolist(),captures=captures,checks=proof,worker=name);details.append(item)
            save(a.out/'branch_progress.json',details)
        print(json.dumps(dict(state=state_id,branches=len(details),future_invalid=sum(x['labels'][-1]<.5 for x in details),seconds=time.perf_counter()-begin)),flush=True)
    np.savez_compressed(a.out/'dataset.npz',**{k:np.asarray(v,np.float32) for k,v in arrays.items()});save(a.out/'records.json',new_rows);save(a.out/'branches.json',details)
    save(a.out/'protocol.json',dict(parent_dataset=str(a.dataset.resolve()),parent_sha256=hashlib.sha256((a.dataset/'dataset.npz').read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256((a.out/'dataset.npz').read_bytes()).hexdigest(),base_states=len(rows),branched_states=len(selected),branches=len(details),future_invalid=sum(d['labels'][-1]<.5 for d in details),train_branches=sum(d['split']=='train' for d in details),validation_branches=sum(d['split']=='validation' for d in details),semantics='Same received input; exact candidate/stop executed without further replanning for 0.8 s; future camera rays follow that candidate with bounded yaw. Privileged geometry used only for labels. Current-map validator and frame timestamps preserved.',limitations='Static ideal geometry, no controller error, finite 0.8 s future observation window, four sampled candidates per selected state; these are supervised branch labels, not complete RL returns.'))

if __name__=='__main__':main()
