"""Two delivered scans + local fused-map evidence. Proposals still require EGO."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from navigation_common import ROOT,CONFIG,save,stationary
from adapted_frontends import inputs,primitives,rotation

ACTION_IDS=[1,2,3,12,15] # 1.5 m at -30/0/+30 degrees, 3 m straight, repair.

def features(cloud,previous,q):
    image,state=inputs(cloud,q);previous=previous or cloud
    prior,_=inputs(previous,q);rot,yaw=rotation(q['goal'],np.array(cloud['origin']))
    cap=cloud.get('capture_state',stationary(cloud['origin']));pc=previous.get('capture_state',stationary(previous['origin']))
    exists=previous['stamp']<cloud['stamp']-1e-9
    context=q['map_context'];grid=np.array(context['grid'],dtype=np.float32).reshape(21,21,6).transpose(2,0,1)
    state[-1]=(CONFIG['handoff']-CONFIG['deadline_guard']-q.get('execution_delay_s',.08)-q.get('context_rpc_ms',0)/1000)/.3
    extra=np.r_[(np.array(previous['origin'])-cloud['origin'])@rot/3,(np.array(pc['v'])-cap['v'])@rot/3,
        (q['t']-previous['stamp'])/2,float(exists),float(q['frontend_backend']=='rog'),np.sin(context['yaw']-yaw),np.cos(context['yaw']-yaw)]
    state=np.r_[state,extra].astype(np.float32);assert state.shape==(33,)
    return np.concatenate([image,prior]),grid,state

def net_class():
    import torch
    class ContextScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rays=torch.nn.Sequential(torch.nn.Conv2d(6,12,3,stride=(1,2),padding=1),torch.nn.ReLU(),torch.nn.Conv2d(12,16,3,stride=2,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d((3,6)),torch.nn.Flatten())
            self.map=torch.nn.Sequential(torch.nn.Conv2d(6,8,3,stride=2,padding=1),torch.nn.ReLU(),torch.nn.Conv2d(8,16,3,stride=2,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d((3,3)),torch.nn.Flatten())
            self.head=torch.nn.Sequential(torch.nn.Linear(288+144+33,64),torch.nn.ReLU(),torch.nn.Linear(64,25))
        def forward(self,rays,local_map,state):return self.head(torch.cat([self.rays(rays),self.map(local_map),state],dim=1)).reshape(-1,5,5)
    return ContextScorer

def predicted_utility(prediction):
    import torch
    return prediction[...,0]-2*torch.sigmoid(prediction[...,1])-.05*prediction[...,2].clamp(min=0)-.08*prediction[...,3].clamp(min=0)+.05*torch.sigmoid(prediction[...,4])

class ContextFrontend:
    requires_context=True
    def __init__(self,weights=None,mode='learned',forced_action=None):
        self.mode=mode;self.forced_action=forced_action;self.reset();self.models=[]
        self.metadata=dict(frontend='context_'+mode,action_ids=ACTION_IDS,rays_shape=[6,17,72],local_map_shape=[6,21,21],state_dim=33,
            age='per 0.1m voxel last fused ray timestamp; NOT occupancy state-update age',decision_margin=.05,ensemble_std_penalty=1.,near_goal_repair_m=3.)
        if weights:
            import torch
            self.torch=torch;torch.set_num_threads(1);self.metadata['weights']=[]
            for weight in weights:
                data=torch.load(weight,map_location='cpu',weights_only=True);model=net_class()().eval();model.load_state_dict(data['state_dict']);self.models.append(model)
                self.metadata['weights'].append(dict(path=str(weight),sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),seed=data['seed'],target=data['target']))
            self.metadata['parameters_per_model']=sum(p.numel() for p in self.models[0].parameters())
            for model in self.models:
                for _ in range(5):
                    with torch.inference_mode():model(torch.zeros(1,6,17,72),torch.zeros(1,6,21,21),torch.zeros(1,33))

    def reset(self):self.scans=[];self.used_force=False
    def observe(self,cloud):
        if self.scans and cloud['stamp']==self.scans[-1]['stamp']:return
        if self.scans and cloud['stamp']<self.scans[-1]['stamp']:raise ValueError('out-of-order frontend observation')
        self.scans.append(cloud);self.scans=self.scans[-2:]

    def propose(self,cloud,q,folder,index):
        begin=time.perf_counter();distance=np.linalg.norm(np.array(q['goal'])-q['start_state']['p'])
        if cloud is None or not q.get('context_ok'):return [],dict(reason='context_unavailable',frontend_ms=0.)
        if not 0<=q['t']-cloud['stamp']<=.5:return [],dict(reason='stale_or_future_cloud',frontend_ms=0.)
        prev=self.scans[-2] if len(self.scans)>1 else None
        rays,local_map,state=features(cloud,prev,q);prepared=time.perf_counter();utility=np.zeros(5);spread=np.zeros(5);raw=None
        if self.models:
            with self.torch.inference_mode():
                raw=self.torch.stack([m(self.torch.from_numpy(rays[None]),self.torch.from_numpy(local_map[None]),self.torch.from_numpy(state[None]))[0] for m in self.models])
                values=predicted_utility(raw);utility=values.mean(dim=0).numpy();spread=values.std(dim=0,unbiased=False).numpy();raw=raw.numpy()
            # Relative benefit gate; ensemble disagreement is an empirical
            # conservative score, not a calibrated safety/confidence bound.
            ranking=utility-spread;action=int(ranking.argmax())
            if ranking[action]<utility[4]+spread[4]+.05:action=4
        elif self.mode=='rule':
            # Sample the four candidate reference lines on the actual local
            # map summary; do not replace the original collision validator.
            routes=primitives(q)
            for i,cid in enumerate(ACTION_IDS[:-1]):
                endpoint=np.array(routes[cid]['points'][-1]);start=np.array(q['start_state']['p']);delta=endpoint-start
                rot,_=rotation(q['goal'],start);points=start+np.linspace(0,1,20)[:,None]*delta;xy=(points-start)@rot
                ix=np.clip(np.rint(xy[:,:2]/.3+10).astype(int),0,20);cells=local_map[:,ix[:,0],ix[:,1]]
                progress=distance-np.linalg.norm(endpoint-np.array(q['goal']))
                utility[i]=progress/np.linalg.norm(delta)-2*cells[1].max()-cells[2].max()-.2*cells[4].mean()
            utility[4]=.8;action=int(utility.argmax())
        else:action=4
        if self.mode=='forced' and not self.used_force:
            action=int(self.forced_action);self.used_force=True
        if distance<3.:action=4
        inferred=time.perf_counter();routes=[] if action==4 else [dict(primitives(q)[ACTION_IDS[action]],score=float(-utility[action]))]
        folder=Path(folder);folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{index:05d}.npz',rays=rays,local_map=local_map,state=state,prediction=raw if raw is not None else np.zeros((0,5,5)))
        record=dict(reason='ok',kind='context',capture_stamp=cloud['stamp'],previous_stamp=(prev or cloud)['stamp'],capture_age_s=q['t']-cloud['stamp'],
            action=action,candidate_id=ACTION_IDS[action],utility=utility.tolist(),disagreement=spread.tolist(),routes=routes,context_rpc_ms=q['context_rpc_ms'],
            preprocess_ms=(prepared-begin)*1000,inference_ms=(inferred-prepared)*1000,observed_fraction=float(rays[2].mean()))
        save(folder/f'{index:05d}.json',record);record['frontend_ms']=(time.perf_counter()-begin)*1000
        return routes,record
