"""Candidate-only frontends. The original EGO solve and validators execute all curves."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from navigation_common import CONFIG,ROOT,save,stationary
from yopo_frontend import YopoFrontend


class AlignedYopo(YopoFrontend):
    """Ablation: same author weights, spatial reference matched to EGO horizon."""
    def __init__(self):
        super().__init__();self.metadata=dict(self.metadata,adaptation='handoff-centred endpoint direction, EGO horizon, near-goal repair at 3 m')

    def propose(self,cloud,request,folder,index):
        begin=time.perf_counter();start=np.array(request['start_state']['p']);goal=np.array(request['goal'])
        if np.linalg.norm(goal-start)<3.:return [],dict(reason='near_goal_repair',frontend_ms=0.)
        routes,info=super().propose(cloud,request,folder,index)
        if info.get('reason')!='ok':return routes,info
        path=Path(folder)/f'{index:05d}.json';record=json.loads(path.read_text());yaw=record['yaw']
        c,s=np.cos(yaw),np.sin(yaw);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        ends=np.asarray(record['decoded_endstates']);scores=np.asarray(record['scores']);routes=[]
        for idx in np.argsort(scores):
            direction=ends[idx,:3]@rot.T;norm=np.linalg.norm(direction)
            if not np.isfinite(direction).all() or norm<1e-5:continue
            endpoint=start+direction/norm*min(CONFIG['planning_horizon'],np.linalg.norm(goal-start))
            routes.append(dict(id=int(idx),score=float(scores[idx]),points=[start.tolist(),endpoint.tolist()]))
            if len(routes)==3:break
        record['original_routes']=record['routes'];record['routes']=routes;record['adaptation']=self.metadata['adaptation'];save(path,record)
        info['frontend_ms']=(time.perf_counter()-begin)*1000
        return routes,info


def rotation(goal,origin):
    direction=np.asarray(goal)-origin;yaw=np.arctan2(direction[1],direction[0]);c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]]),yaw


def inputs(cloud,request):
    origin=np.asarray(cloud['origin']);rot,yaw=rotation(request['goal'],origin)
    points=np.asarray(cloud['points']);r=np.linalg.norm(points-origin,axis=1).reshape(17,-1)
    if r.shape[1]!=720:raise ValueError('requires unchanged 17 x 720 ray sensor')
    # Circular panorama centred on the capture-to-goal direction; conservative
    # min pooling within each five-degree bin, never interpolate free space.
    indices=(np.arange(720)+int(np.rint(yaw/(2*np.pi)*720))-360)%720
    r=r[:,indices].reshape(17,72,10)
    valid=np.isfinite(r)&(r>0);ranges=np.min(np.where(valid,r,0),axis=-1)
    mask=valid.all(axis=-1);hit=(ranges<=CONFIG['sensor_range'])&mask
    image=np.stack([np.minimum(ranges,CONFIG['sensor_range'])/CONFIG['sensor_range'],hit.astype(float),mask.astype(float)]).astype(np.float32)
    cap=cloud.get('capture_state',stationary(origin));handoff=request['start_state'];goal=np.array(request['goal'])-handoff['p'];distance=np.linalg.norm(goal)
    age=request['t']-cloud['stamp'];budget=CONFIG['handoff']-CONFIG['deadline_guard']-.08
    state=np.concatenate([np.array(cap['v'])@rot/3,np.array(cap['a'])@rot/6,np.array(handoff['v'])@rot/3,np.array(handoff['a'])@rot/6,
        (np.array(handoff['p'])-origin)@rot/3,goal@rot/max(distance,1e-6),[distance/20,age/.5,(age+CONFIG['handoff'])/.8,budget/.3]]).astype(np.float32)
    assert state.shape==(22,)
    return image,state


def primitives(request):
    start=np.array(request['start_state']['p']);goal=np.array(request['goal']);rot,_=rotation(goal,start)
    distance=np.linalg.norm(goal-start);routes=[]
    for length in [1.5,2.25,3.]:
        for degrees in [-60,-30,0,30,60]:
            angle=np.deg2rad(degrees);direction=np.array([np.cos(angle),np.sin(angle),0.])@rot.T
            endpoint=start+min(length,distance)*direction
            endpoint[2]=start[2]+np.clip(goal[2]-start[2],-.25,.25)
            routes.append(dict(id=len(routes),points=[start.tolist(),endpoint.tolist()]))
    return routes


def model_class():
    import torch
    class RangeScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=torch.nn.Sequential(torch.nn.Conv2d(3,12,3,stride=(1,2),padding=1),torch.nn.ReLU(),
                torch.nn.Conv2d(12,24,3,stride=(2,2),padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d((3,9)),torch.nn.Flatten())
            self.head=torch.nn.Sequential(torch.nn.Linear(24*3*9+22,64),torch.nn.ReLU(),torch.nn.Linear(64,32))
        def forward(self,image,state):
            out=self.head(torch.cat([self.encoder(image),state],dim=1));return out[:,:16],out[:,16:]
    return RangeScorer


class RangeFrontend:
    def __init__(self,weight=None):
        self.rule=weight is None;self.metadata=dict(frontend='rule_primitives' if self.rule else 'learned_range_scorer',input_shape=[3,17,72],state_dim=22,
            candidates='15 straight references: 3 lengths x 5 headings; repair action 15',near_goal_repair_m=3.,extra_delay_assumption_s=.08)
        if not self.rule:
            import torch
            self.torch=torch;torch.set_num_threads(1);self.model=model_class()().eval()
            data=torch.load(weight,map_location='cpu',weights_only=True);self.model.load_state_dict(data['state_dict'])
            self.metadata.update(weight=str(weight),sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),parameters=sum(p.numel() for p in self.model.parameters()),training_seed=data['seed'])
            for _ in range(5):
                with torch.inference_mode():self.model(torch.zeros(1,3,17,72),torch.zeros(1,22))

    def propose(self,cloud,request,folder,index):
        begin=time.perf_counter();start=np.array(request['start_state']['p']);goal=np.array(request['goal'])
        if cloud is None:return [],dict(reason='no_received_cloud',frontend_ms=0.)
        if request['t']-cloud['stamp']>.5:return [],dict(reason='stale_received_cloud',frontend_ms=0.)
        if np.linalg.norm(goal-start)<3.:return [],dict(reason='near_goal_repair',frontend_ms=0.)
        image,state=inputs(cloud,request);routes=primitives(request);prepared=time.perf_counter()
        if self.rule:
            # Same candidate family; rank by goal alignment and observed radial
            # clearance. This is a heuristic proposal score, never a safety test.
            scores=[]
            radial=image[0]*CONFIG['sensor_range']
            for route in routes:
                delta=np.array(route['points'][-1])-start;angle=np.arctan2(delta[1],delta[0]);_,yaw=rotation(goal,cloud['origin'])
                column=int(np.rint(((angle-yaw+np.pi)%(2*np.pi))/(2*np.pi)*72))%72
                sector=radial[5:12][:,(np.arange(column-2,column+3)%72)]
                free=float(sector.min());length=np.linalg.norm(delta);progress=np.linalg.norm(goal-start)-np.linalg.norm(goal-np.array(route['points'][-1]))
                scores.append(progress/max(length,1e-6)+.1*length-2*max(0.,length+.6-free))
            scores=np.asarray(scores+[.0]);feasible=np.ones(16)
        else:
            with self.torch.inference_mode():score,logit=self.model(self.torch.from_numpy(image[None]),self.torch.from_numpy(state[None]))
            scores=score.numpy()[0];feasible=self.torch.sigmoid(logit).numpy()[0]
        inferred=time.perf_counter()
        # Predicted feasibility is only a ranking feature. All accepted routes
        # still receive original map, EGO, full spline and backup validation.
        utility=scores-(1-feasible)*2.;order=np.argsort(-utility);chosen=[]
        for idx in order:
            if idx==15:break
            chosen.append(dict(routes[int(idx)],score=float(-utility[idx])))
            if len(chosen)==3:break
        record=dict(reason='ok',capture_stamp=cloud['stamp'],capture_age_s=request['t']-cloud['stamp'],capture_origin=cloud['origin'],capture_state=cloud.get('capture_state'),
            handoff_state=request['start_state'],scores=scores.tolist(),predicted_feasibility=feasible.tolist(),order=order.tolist(),routes=chosen,
            observed_fraction=float(image[2].mean()),preprocess_ms=(prepared-begin)*1000,inference_ms=(inferred-prepared)*1000,kind=self.metadata['frontend'])
        folder=Path(folder);folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{index:05d}.npz',range_image=image,state=state)
        save(folder/f'{index:05d}.json',record);record['frontend_ms']=(time.perf_counter()-begin)*1000
        return chosen,{k:record[k] for k in ['reason','capture_stamp','capture_age_s','observed_fraction','preprocess_ms','inference_ms','frontend_ms','kind']}
