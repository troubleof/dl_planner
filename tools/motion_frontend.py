"""State-conditioned directional risk. No online trajectory search or risk rollout."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from navigation_common import CONFIG,ROOT,RADIUS,Curve,save
from adapted_frontends import rotation
from direct_frontend import observation as original_observation,net_class as original_net,decode,STATE,K,PARAMS

SPEC=json.loads((ROOT/'config/motion_risk.json').read_text())
FORMAT=SPEC['format']

def directional(points,p,v,a):
    delta=np.asarray(points)-np.asarray(p);distance=np.linalg.norm(delta,axis=-1)
    direction=delta/np.maximum(distance[...,None],1e-8)
    closing=np.maximum(0,np.sum(direction*np.asarray(v),axis=-1))
    acceleration=np.maximum(0,np.sum(direction*np.asarray(a),axis=-1))
    tau=SPEC['reaction_allowance_s'];b=SPEC['braking_reference_mps2']
    travel=closing*tau+.5*acceleration*tau*tau+(closing+acceleration*tau)**2/(2*b)
    return closing,acceleration,distance-RADIUS-travel

def observation(cloud,q,backend,channels=6):
    if cloud.get('sensor')=='d435i':
        from d435i_adapter import camera_observation
        return camera_observation(cloud,q,backend,channels)
    image,state=original_observation(cloud,q,backend)
    if channels==3:return image,state
    points=np.asarray(cloud['points']).reshape(17,720,3)
    origin=np.asarray(cloud['origin']);_,yaw=rotation(q['goal'],origin)
    indices=(np.arange(720)+int(np.rint(yaw/(2*np.pi)*720))-360)%720
    points=points[:,indices].reshape(17,72,10,3)
    ranges=np.linalg.norm(points-origin,axis=-1)
    hits=np.isfinite(ranges)&(ranges>0)&(ranges<=CONFIG['sensor_range'])
    s=q['start_state'];closing,acceleration,margin=directional(points,s['p'],s['v'],s['a'])
    # Compute all ten observed rays before pooling: the nearest point need not
    # be the highest-risk point. No-return endpoints are never obstacles.
    vel=np.max(np.where(hits,closing/3,0),axis=-1)
    acc=np.max(np.where(hits,acceleration/6,0),axis=-1)
    safe_margin=np.min(np.where(hits,margin,SPEC['margin_scale_m']),axis=-1)
    safe_margin=np.where(hits.any(axis=-1),np.clip(safe_margin/SPEC['margin_scale_m'],-1,1),0)
    extra=np.stack([vel,acc,safe_margin]).astype(np.float32)
    extra*=image[2:3]  # Invalid bins do not acquire artificial risk evidence.
    return np.concatenate([image,extra]),state

def net_class(channels=6):
    import torch
    class MotionNetwork(original_net()):
        def __init__(self):
            super().__init__()
            if channels!=3:self.encoder[0]=torch.nn.Conv2d(channels,12,3,stride=(1,2),padding=1)
    return MotionNetwork

def cloud_risk(plan,cloud):
    """Offline only. Observed-point risk is auxiliary, never map admission."""
    points=np.asarray(cloud['points']);ranges=np.linalg.norm(points-np.asarray(cloud['origin']),axis=1)
    hits=points[(ranges>0)&(ranges<=CONFIG['sensor_range'])]
    if not len(hits):return dict(risk=0.,min_margin_m=SPEC['margin_scale_m'],observed_hits=0)
    # Voxel representatives reduce offline work; this is not a collision test.
    _,ix=np.unique(np.floor(hits/.1).astype(np.int32),axis=0,return_index=True);hits=hits[ix]
    c=Curve(plan);t=np.linspace(0,min(c.duration,SPEC['label_horizon_s']),SPEC['label_samples'])
    minimum=SPEC['margin_scale_m']
    for p,v,a in zip(c.s(t),c.v(t),c.a(t)):
        _,_,margin=directional(hits,p,v,a);minimum=min(minimum,float(margin.min()))
    risk=float(np.clip((.5-minimum)/1.5,0,1))
    return dict(risk=risk,min_margin_m=minimum,observed_hits=len(hits))

class MotionFrontend:
    direct_trajectory=True
    fallback_repair=False
    def __init__(self,backend,weight,plan_budget_ms=None,frontend_budget_ms=None):
        import torch
        self.torch=torch;torch.set_num_threads(1);self.backend=backend
        self.plan_budget_ms=SPEC['plan_budget_ms'] if plan_budget_ms is None else plan_budget_ms
        self.frontend_budget_ms=SPEC['frontend_budget_ms'] if frontend_budget_ms is None else frontend_budget_ms
        ckpt=torch.load(weight,map_location='cpu',weights_only=True);self.channels=ckpt['channels']
        self.sensor=ckpt.get('sensor','panorama');self.input_shape=ckpt.get('input_shape',[self.channels,17,72])
        self.model=net_class(self.channels)().eval();self.model.load_state_dict(ckpt['state_dict'])
        self.metadata=dict(frontend=FORMAT,weight=str(weight),sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),
            variant=ckpt['variant'],seed=ckpt['seed'],channels=self.channels,parameters=sum(p.numel() for p in self.model.parameters()),
            plan_budget_ms=self.plan_budget_ms,frontend_budget_ms=self.frontend_budget_ms,online_optimizer=False,
            online_rollout=False,risk_spec=SPEC,sensor=self.sensor,input_shape=self.input_shape,logging='full proposal evidence; included in measured RPC')
        for _ in range(10):
            with torch.inference_mode():self.model(torch.zeros(1,*self.input_shape),torch.zeros(1,STATE))

    def compute(self,cloud,q):
        begin=time.perf_counter()
        if self.sensor!=cloud.get('sensor','panorama'):raise ValueError('camera/panorama checkpoint mismatch')
        image,state=observation(cloud,q,self.backend,self.channels)
        if list(image.shape)!=self.input_shape:raise ValueError('checkpoint image shape mismatch')
        prepared=time.perf_counter()
        with self.torch.inference_mode():raw,pred=self.model(self.torch.from_numpy(image[None]),self.torch.from_numpy(state[None]))
        raw,pred=raw.numpy()[0],pred.numpy()[0];inferred=time.perf_counter()
        if not np.isfinite(raw).all() or not np.isfinite(pred).all():raise ValueError('nonfinite prediction')
        plans,utility=decode(q,raw,pred);decoded=time.perf_counter()
        return plans,dict(range_image=image,state=state,timed_parameters=raw,prediction=pred),dict(
            preprocess_ms=(prepared-begin)*1000,inference_ms=(inferred-prepared)*1000,decode_ms=(decoded-inferred)*1000,
            compute_ms=(decoded-begin)*1000,utility=utility.tolist())

    def propose(self,cloud,q,folder,index):
        begin=time.perf_counter()
        def reject(why):return [],dict(reason=why,frontend_ms=(time.perf_counter()-begin)*1000)
        if cloud is None:return reject('no_received_cloud')
        if not 0<=q['t']-cloud['stamp']<=.5:return reject('stale_or_future_cloud')
        try:plans,arrays,record=self.compute(cloud,q)
        except (ValueError,KeyError,TypeError):return reject('invalid_observation')
        if record['compute_ms']>self.frontend_budget_ms:plans=[];reason='frontend_budget'
        else:reason='ok'
        folder=Path(folder);folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'{index:05d}.npz',**arrays)
        record.update(reason=reason,kind=FORMAT,capture_stamp=cloud['stamp'],capture_age_s=q['t']-cloud['stamp'],direct_plans=plans)
        save(folder/f'{index:05d}.json',record)
        record['frontend_ms']=(time.perf_counter()-begin)*1000
        return plans,record
