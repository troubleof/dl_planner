"""Bounded direct candidates with learned future-map schedule survival.

Three small architectures share the same seeds, exact decoder and hard checker.
Predicted survival only ranks candidates; it never authorizes execution.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from navigation_common import ROOT,save
from visible_frontend import seeds
from direct_frontend import decode
from motion_frontend import observation

FORMAT='d435i_recovery_survival_v1'
HORIZONS=[.2,.4,.8]
FEATURES=22
STATE=38
K=45

def candidate_bank(q):
    seed,features=seeds(q)
    if seed is None:return None,None
    raw=[];feats=[]
    for shift in [-.18,0.,.18]:
        r=seed.copy();r[:,0]=np.clip(r[:,0]+shift,np.log(.6),np.log(5.95));f=features.copy();f[:,3]=np.exp(r[:,0])/6;raw.append(r);feats.append(f)
    return np.concatenate(raw),np.concatenate(feats)

def rule_values(q,seed):
    from navigation_common import Curve
    plans,_=decode(q,seed,np.column_stack([np.zeros(len(seed)),np.full(len(seed),20.)]));goal=np.asarray(q.get('local_progress_target',q['goal']));d=np.linalg.norm(goal-q['start_state']['p']);score=np.zeros(len(seed),np.float32)
    for entry in plans:
        c=Curve(entry['plan']);early=(d-np.linalg.norm(goal-c.s(min(.6,c.duration))))/.6;end=d-np.linalg.norm(goal-c.s(c.duration));score[entry['id']]=early+.12*end-.03*c.duration
    return score

def inputs(cloud,previous,q,backend,seed,features,map_features):
    image,state=observation(cloud,q,backend,6)
    history=np.zeros_like(image);meta=np.zeros(6,np.float32)
    if previous is not None and 0<cloud['stamp']-previous['stamp']<=1.:
        history,_=observation(previous,q,backend,6)
        R=np.asarray(cloud['camera_to_world']);old=np.asarray(previous['camera_to_world']);relative=R.T@old
        meta=np.r_[(np.asarray(previous['origin'])-cloud['origin'])@R/3,relative[2,2],relative[0,2],(cloud['stamp']-previous['stamp'])].astype(np.float32)
    rule=rule_values(q,seed)
    f=np.column_stack([features,np.asarray(map_features),rule]).astype(np.float32)
    return image,history,np.r_[state,meta].astype(np.float32),f,rule

def network_class(kind):
    import torch
    class Survival(torch.nn.Module):
        def __init__(self):
            super().__init__();self.kind=kind
            if kind!='mlp':
                self.encoder=torch.nn.Sequential(torch.nn.Conv2d(6,12,3,stride=(2,2),padding=1),torch.nn.ReLU(),torch.nn.Conv2d(12,16,3,stride=2,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d((3,5)),torch.nn.Flatten())
            if kind=='temporal':self.gate=torch.nn.Linear(STATE,240)
            n=STATE+(0 if kind=='mlp' else 480 if kind=='temporal' else 240)
            self.fusion=torch.nn.Sequential(torch.nn.Linear(n,64),torch.nn.ReLU())
            self.head=torch.nn.Sequential(torch.nn.Linear(64+FEATURES,64),torch.nn.ReLU(),torch.nn.Linear(64,4))
        def forward(self,image,previous,state,features):
            z=state
            if self.kind!='mlp':
                now=self.encoder(image)
                if self.kind=='temporal':
                    old=self.encoder(previous);difference=(now-old)*torch.sigmoid(self.gate(state));z=torch.cat([state,now,difference],dim=-1)
                else:z=torch.cat([state,now],dim=-1)
            z=self.fusion(z)[:,None].expand(-1,features.shape[1],-1)
            return self.head(torch.cat([z,features],dim=-1))
    return Survival

def probabilities(logits):
    # P(current feasible) followed by conditional survival hazards. Cumulative
    # survival cannot increase with horizon, even if the current map later clears.
    return np.cumprod(1/(1+np.exp(-np.clip(logits,-30,30))),axis=-1)

class RecoveryFrontend:
    visible_context=True
    direct_trajectory=True
    fallback_repair=False
    plan_budget_ms=100.
    def __init__(self,backend,weight=None,plain_rule=False):
        self.backend=backend;self.rule=weight is None;self.plain_rule=plain_rule;self.reset()
        self.metadata=dict(frontend=FORMAT,rule=self.rule,plain_rule=plain_rule,input_shape=[6,60,106],state_dim=STATE,candidate_features=FEATURES,horizons=HORIZONS,online_optimizer=False,risk_weight=.8,geometry='15 visible seeds times three fixed log-duration shifts [-.18,0,.18]; top15 checked',label='future received map schedule survival, not a certificate')
        if weight:
            import torch
            self.torch=torch;torch.set_num_threads(1);ck=torch.load(weight,map_location='cpu',weights_only=True);assert ck['format']==FORMAT
            self.current_only=ck.get('current_only',False);self.model=network_class(ck['kind'])().eval();self.model.load_state_dict(ck['state_dict']);self.metadata.update(weight=str(weight),kind=ck['kind'],seed=ck['seed'],current_only=self.current_only,sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),parameters=sum(p.numel() for p in self.model.parameters()))
            with torch.inference_mode():
                for _ in range(5):self.model(torch.zeros(1,6,60,106),torch.zeros(1,6,60,106),torch.zeros(1,STATE),torch.zeros(1,K,FEATURES))
    def reset(self):self.latest=None;self.previous=None
    def observe(self,cloud):
        if self.latest is None or cloud['stamp']>self.latest['stamp']+1e-9:self.previous=self.latest;self.latest=cloud
    def propose(self,cloud,q,folder,index):
        begin=time.perf_counter()
        def reject(reason):return [],dict(reason=reason,frontend_ms=(time.perf_counter()-begin)*1000)
        if not q.get('context_ok'):return reject('visible_context_failed')
        q['local_progress_target']=q['map_context']['waypoint']
        if cloud is None or not 0<=q['t']-cloud['stamp']<=.5:return reject('stale_or_missing_cloud')
        seed,features=candidate_bank(q)
        if seed is None:return reject('no_known_start')
        plans,_=decode(q,seed,np.column_stack([np.zeros(K),np.full(K,20.)]));feature_ms=0.
        map_features=np.zeros((K,15),np.float32)
        for i in range(0,K,15):
            result=self.worker.request(dict(op='candidate_features',t=q['t'],direct_plans=plans[i:i+15]));feature_ms+=result['rpc_ms']
            for entry in result['features']:map_features[entry['id']]=entry['values']
        image,previous,state,features,rule=inputs(cloud,self.previous,q,self.backend,seed,features,map_features)
        prepared=time.perf_counter()
        if self.rule:
            risk=map_features[:,0::3].mean(axis=1)+map_features[:,1::3].mean(axis=1)
            logits=np.zeros((K,4),np.float32);survival=np.ones((K,4),np.float32);utility=rule if self.plain_rule else rule-.8*risk
        else:
            with self.torch.inference_mode():logits=self.model(*(self.torch.from_numpy(x[None]) for x in [image,previous,state,features])).numpy()[0]
            survival=probabilities(logits);utility=rule-.8*(1-survival[:,0 if self.current_only else 3])
        inferred=time.perf_counter();pred=np.column_stack([utility,np.full(K,20.)]);ranked,_=decode(q,seed,pred);plans=[];seen=set()
        for entry in ranked:
            key=tuple(np.round(seed[entry['id']],6))
            if key in seen:continue
            seen.add(key);plans.append(entry)
            if len(plans)==15:break
        # Nested context RPCs finish before the enclosing plan log is written.
        # Its final sequential log index is the worker's current request id.
        index=self.worker.request_id
        folder=Path(folder);folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'{index:05d}.npz',image=image,previous=previous,state=state,features=features,seed=seed,logits=logits,survival=survival,prediction=pred,map_features=map_features)
        record=dict(reason='ok',kind=FORMAT,capture_stamp=cloud['stamp'],previous_stamp=None if self.previous is None else self.previous['stamp'],direct_plans=plans,preprocess_ms=(prepared-begin)*1000,inference_ms=(inferred-prepared)*1000,candidate_features_rpc_ms=feature_ms)
        save(folder/f'{index:05d}.json',record);record['frontend_ms']=(time.perf_counter()-begin)*1000
        return plans,record
