"""Short, PVA-continuous direct splines conditioned on known-free guidance."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from navigation_common import ROOT,CONFIG,save,Curve
from adapted_frontends import rotation
from direct_frontend import BASE,decode,controls_torch
from motion_frontend import observation

FORMAT='d435i_visible_short_v1'
K=15

def seeds(q):
    s=q['start_state'];p=np.asarray(s['p']);v=np.asarray(s['v']);a=np.asarray(s['a']);rot,_=rotation(q['goal'],p)
    context=q.get('map_context') or {};ends=np.asarray(context.get('endpoints',[]))
    if ends.shape!=(15,3):return None,None
    raw=np.zeros((K,15),np.float32);delta=ends-p;local=delta@rot;distance=np.linalg.norm(delta,axis=1)
    duration=np.maximum(np.maximum(.8,2*distance/1.5),2*np.linalg.norm(v)/2.5);vl=v@rot;al=a@rot
    for _ in range(12):
        duration=np.minimum(duration,5.9);h=duration[:,None]/9
        free=np.einsum('ij,kjl->kil',BASE,np.stack([local,duration[:,None]*vl,duration[:,None]**2*al],axis=1))
        controls=np.concatenate([np.stack([-vl*h+al*h*h/3,-al*h*h/6,vl*h+al*h*h/3],axis=1),free,np.repeat(local[:,None],3,axis=1)],axis=1)
        vc=np.diff(controls,axis=1)/h[:,:,None];ac=np.diff(vc,axis=1)/h[:,:,None]
        bad=(np.linalg.norm(vc,axis=-1).max(axis=1)>2.8)|(np.linalg.norm(ac,axis=-1).max(axis=1)>5.5)
        if not bad.any():break
        duration=np.where(bad,duration*1.18,duration)
    duration=np.minimum(duration,5.9);raw[:,0]=np.log(duration);raw[:,13:15]=local[:,:2]/3
    target=np.asarray(context.get('waypoint',q['goal']));progress=np.linalg.norm(target-p)-np.linalg.norm(target-ends,axis=1)
    features=np.column_stack([local/3,duration/6,progress/3,distance/3]).astype(np.float32)
    return raw,features

def network_class():
    import torch
    class VisibleNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Sequential(torch.nn.Conv2d(6,12,3,stride=(1,2),padding=1),torch.nn.ReLU(),torch.nn.Conv2d(12,24,3,stride=2,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d((3,9)),torch.nn.Flatten())
            self.fusion=torch.nn.Sequential(torch.nn.Linear(648+32,64),torch.nn.ReLU())
            self.geometry=torch.nn.Sequential(torch.nn.Linear(64+6,32),torch.nn.ReLU(),torch.nn.Linear(32,15))
            self.scorer=torch.nn.Sequential(torch.nn.Linear(64+6+15,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
            torch.nn.init.zeros_(self.geometry[-1].weight);torch.nn.init.zeros_(self.geometry[-1].bias)
        def forward(self,image,state,seed,features):
            z=self.fusion(torch.cat([self.encoder(image),state],dim=-1))[:,None].expand(-1,K,-1)
            delta=self.geometry(torch.cat([z,features],dim=-1));duration=(seed[...,:1]+.2*torch.tanh(delta[...,:1])).clamp(np.log(.6),np.log(5.95))
            raw=torch.cat([duration,.02*torch.tanh(delta[...,1:13]),seed[...,13:15]*(1+.15*torch.tanh(delta[...,13:15]))],dim=-1)
            score=self.scorer(torch.cat([z,features,raw],dim=-1));return raw,score
    return VisibleNetwork

class VisibleFrontend:
    visible_context=True
    direct_trajectory=True
    fallback_repair=False
    plan_budget_ms=100.
    def __init__(self,backend,weight=None,hybrid=False):
        self.backend=backend;self.rule=weight is None;self.hybrid=hybrid
        self.metadata=dict(frontend=FORMAT,rule=self.rule,hybrid=hybrid,input_shape=[6,60,106],state_dim=32,context='current fused known-free grid, 8 ms Dijkstra budget',max_candidates=15,online_optimizer=False)
        if weight:
            import torch
            self.torch=torch;torch.set_num_threads(1);ck=torch.load(weight,map_location='cpu',weights_only=True);assert ck['format']==FORMAT
            self.model=network_class()().eval();self.model.load_state_dict(ck['state_dict'])
            self.metadata.update(weight=str(weight),sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),seed=ck['seed'],parameters=sum(p.numel() for p in self.model.parameters()))
            for _ in range(5):
                with torch.inference_mode():self.model(torch.zeros(1,6,60,106),torch.zeros(1,32),torch.zeros(1,15,15),torch.zeros(1,15,6))
    def propose(self,cloud,q,folder,index):
        begin=time.perf_counter()
        def reject(reason):return [],dict(reason=reason,frontend_ms=(time.perf_counter()-begin)*1000)
        if not q.get('context_ok'):return reject('visible_context_failed')
        q['local_progress_target']=q['map_context']['waypoint']
        if cloud is None or not 0<=q['t']-cloud['stamp']<=.5:return reject('stale_or_missing_cloud')
        image,state=observation(cloud,q,self.backend,6);seed,features=seeds(q)
        if seed is None:return reject('no_known_start')
        prepared=time.perf_counter()
        if self.rule:
            raw=seed;pred=rule_prediction(q,seed)
        else:
            with self.torch.inference_mode():r,p=self.model(self.torch.from_numpy(image[None]),self.torch.from_numpy(state[None]),self.torch.from_numpy(seed[None]),self.torch.from_numpy(features[None]))
            raw,pred=r.numpy()[0],p.numpy()[0]
        inferred=time.perf_counter();plans,utility=decode(q,raw,pred)
        if self.hybrid:
            fixed,_=decode(q,seed,rule_prediction(q,seed))
            plans=plans[:8]+[dict(c,id=c['id']+15) for c in fixed[:7]]
        folder=Path(folder);folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'{index:05d}.npz',range_image=image,state=state,seed=seed,features=features,timed_parameters=raw,prediction=pred)
        record=dict(reason='ok',kind=FORMAT,capture_stamp=cloud['stamp'],direct_plans=plans,utility=utility.tolist(),preprocess_ms=(prepared-begin)*1000,inference_ms=(inferred-prepared)*1000)
        save(folder/f'{index:05d}.json',record);record['frontend_ms']=(time.perf_counter()-begin)*1000
        return plans,record

def rule_prediction(q,seed):
    pred=np.column_stack([np.zeros(15),np.full(15,20.)]);plans,_=decode(q,seed,pred);goal=np.asarray(q.get('local_progress_target',q['goal']));initial=np.linalg.norm(goal-q['start_state']['p'])
    for entry in plans:
        c=Curve(entry['plan']);early=(initial-np.linalg.norm(goal-c.s(min(.6,c.duration))))/.6;end=initial-np.linalg.norm(goal-c.s(c.duration))
        pred[entry['id'],0]=early+.12*end-.03*c.duration
    return pred
