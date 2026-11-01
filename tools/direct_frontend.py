"""Timed cubic splines. No online trajectory fitting, retiming or EGO optimization."""
import hashlib,time
from pathlib import Path
import numpy as np
from navigation_common import CONFIG,save
from adapted_frontends import inputs,rotation

FORMAT='direct_timed_bspline_v2'
N=12
K=15
PARAMS=15  # log duration, six planar residuals and terminal XY / 3 m
STATE=32

def base_matrix():
    # Fixed offline linear conversion from minimum-jerk polynomial to this basis.
    # Computed once at module import; no fitting/search runs during plan requests.
    from scipy.interpolate import BSpline
    u=np.linspace(0,1,145);basis=BSpline(np.arange(16)-3,np.eye(12),3)(u*9)
    known=np.zeros((12,3));known[0]=[0,-1/9,1/243];known[1]=[0,0,-1/486];known[2]=[0,1/9,1/243];known[9:]=[1,0,0]
    polynomial=np.stack([10*u**3-15*u**4+6*u**5,u-6*u**3+8*u**4-3*u**5,.5*u**2-1.5*u**3+1.5*u**4-.5*u**5],axis=1)
    return np.linalg.pinv(basis[:,3:9])@(polynomial-basis@known)
BASE=base_matrix()

def observation(cloud,q,backend):
    xyz=np.asarray(cloud['points'])
    if xyz.shape!=(17*720,3) or not np.isfinite(xyz).all():raise ValueError('invalid scan')
    image,state=inputs(dict(cloud,points=xyz),q)
    s=q['start_state'];rot,_=rotation(q['goal'],np.asarray(s['p']))
    state=np.r_[state,float(backend=='rog'),np.asarray(s['v'])@rot/3,np.asarray(s['a'])@rot/6,
                (np.asarray(q['goal'])-s['p'])@rot/20].astype(np.float32)
    if not np.isfinite(image).all() or not np.isfinite(state).all():raise ValueError('nonfinite observation')
    return image,state

def canonical(raw,state):
    import torch
    # Goal capture is an explicit decoder condition; no map or optimizer is used.
    duration=raw[...,:1].clamp(np.log(.6),np.log(5.95))
    goal=state[:,29:32]*20/3;distance=torch.linalg.vector_norm(goal*3,dim=1)
    endpoint=raw[...,13:15]
    limits=torch.tensor([1.5]*5+[2.25]*5+[3.]*5,device=raw.device,dtype=raw.dtype)[None,:]
    limits=torch.minimum(limits,distance[:,None])/3
    endpoint=endpoint*torch.minimum(torch.ones_like(limits),limits/endpoint.norm(dim=-1).clamp_min(1e-6))[:,:,None]
    central=torch.tensor([i%5==2 for i in range(K)],device=raw.device)[None,:,None]
    endpoint=torch.where((distance<1.5)[:,None,None]&central,goal[:,None,:2].expand(-1,K,-1),endpoint)
    residual=raw[...,1:13].clamp(-.2,.2)*(distance/1.5).clamp(max=1).square()[:,None,None]
    return torch.cat([duration,residual,endpoint],dim=-1)

def net_class():
    import torch
    class DirectNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=torch.nn.Sequential(torch.nn.Conv2d(3,12,3,stride=(1,2),padding=1),torch.nn.ReLU(),
                torch.nn.Conv2d(12,24,3,stride=2,padding=1),torch.nn.ReLU(),torch.nn.AdaptiveAvgPool2d((3,9)),torch.nn.Flatten())
            self.fusion=torch.nn.Sequential(torch.nn.Linear(648+STATE,64),torch.nn.ReLU())
            self.geometry=torch.nn.Linear(64,K*PARAMS)
            self.scorer=torch.nn.Sequential(torch.nn.Linear(64+PARAMS+K,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
        def forward(self,image,state):
            z=self.fusion(torch.cat([self.encoder(image),state],dim=1))
            raw=canonical(self.geometry(z).reshape(-1,K,PARAMS),state)
            identity=torch.eye(K,device=z.device,dtype=z.dtype)[None].expand(z.shape[0],-1,-1)
            score=self.scorer(torch.cat([z[:,None].expand(-1,K,-1),raw,identity],dim=-1))
            return raw,score
    return DirectNetwork

def controls_torch(raw,state):
    import torch
    dt=raw[...,0].exp()/9
    v=state[:,None,23:26]*3;a=state[:,None,26:29]*6;h=dt[...,None]
    q0=-v*h+a*h*h/3;q1=-a*h*h/6;q2=v*h+a*h*h/3
    end=torch.cat([raw[...,13:15]*3,(state[:,None,31:32]*20).clamp(-.25,.25).expand(-1,K,-1)],dim=-1)
    T=dt[...,None]*9;coef=torch.tensor(BASE,device=raw.device,dtype=raw.dtype)
    free=end[:,:,None,:]*coef[None,None,:,0:1]+(v*T)[:,:,None,:]*coef[None,None,:,1:2]+(a*T*T)[:,:,None,:]*coef[None,None,:,2:3]
    residual=raw[...,1:13].reshape(-1,K,6,2)*3
    free=free+torch.cat([residual,torch.zeros_like(residual[...,:1])],dim=-1)
    controls=torch.cat([torch.stack([q0,q1,q2],dim=2),free,end[:,:,None,:].expand(-1,-1,3,-1)],dim=2)
    return controls,dt

def decode(q,raw,prediction):
    start=np.asarray(q['start_state']['p']);v=np.asarray(q['start_state']['v']);a=np.asarray(q['start_state']['a'])
    rot,_=rotation(q['goal'],start)
    probability=1/(1+np.exp(-np.clip(prediction[:,1],-30,30)))
    utility=prediction[:,0]-2*(1-probability)
    candidates=[]
    for index in np.argsort(-utility,kind='stable'):
        duration=float(np.exp(raw[index,0]));dt=duration/9
        endpoint=np.r_[raw[index,13:15].astype(float)*3,np.clip((np.asarray(q['goal'])-start)[2],-.25,.25)]
        free=BASE@np.stack([endpoint,v@rot*duration,a@rot*duration**2])
        free[:,:2]+=raw[index,1:13].reshape(6,2)*3
        points=np.vstack([free,endpoint])@rot.T+start
        controls=np.vstack([start-v*dt+a*dt*dt/3,start-a*dt*dt/6,start+v*dt+a*dt*dt/3,
                            points[:6],np.repeat(points[-1:],3,axis=0)])
        candidates.append(dict(id=int(index),score=float(utility[index]),plan=dict(start=q['t']+CONFIG['handoff'],dt=dt,
            duration=9*dt,controls=controls.tolist())))
    return candidates,utility

class DirectFrontend:
    direct_trajectory=True
    def __init__(self,backend,weight,fallback_repair=False):
        import torch
        self.torch=torch;torch.set_num_threads(1);self.backend=backend;self.fallback_repair=fallback_repair
        checkpoint=torch.load(weight,map_location='cpu',weights_only=True)
        assert checkpoint['format']==FORMAT
        self.model=net_class()().eval();self.model.load_state_dict(checkpoint['state_dict'])
        self.metadata=dict(frontend=FORMAT,backend=backend,weight=str(weight),sha256=hashlib.sha256(Path(weight).read_bytes()).hexdigest(),
            seed=checkpoint['seed'],parameters=sum(p.numel() for p in self.model.parameters()),controls=N,candidates=K,
            fallback_repair=fallback_repair,input_shape=[3,17,72],state_dim=STATE,duration_bounds=[.6,5.95],
            exact_initial_pva=True,terminal_rest=True,near_goal_central_endpoint_m=1.5,parameterization='planar residual + analytic vertical/minimum-jerk base; fixed linear spline conversion',
            online_optimization=False,score_labels='validator and backup check of the actual frozen network output')
        for _ in range(5):
            with torch.inference_mode():self.model(torch.zeros(1,3,17,72),torch.zeros(1,STATE))
    def propose(self,cloud,q,folder,index):
        begin=time.perf_counter()
        def reject(reason):return [],dict(reason=reason,frontend_ms=(time.perf_counter()-begin)*1000)
        if cloud is None:return reject('no_received_cloud')
        if not 0<=q['t']-cloud['stamp']<=.5:return reject('stale_or_future_cloud')
        try:image,state=observation(cloud,q,self.backend)
        except (ValueError,KeyError,TypeError):return reject('invalid_observation')
        prepared=time.perf_counter()
        with self.torch.inference_mode():raw,pred=self.model(self.torch.from_numpy(image[None]),self.torch.from_numpy(state[None]))
        raw,pred=raw.numpy()[0],pred.numpy()[0];inferred=time.perf_counter()
        if not np.isfinite(raw).all() or not np.isfinite(pred).all():return reject('nonfinite_prediction')
        plans,utility=decode(q,raw,pred)
        folder=Path(folder);folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{index:05d}.npz',range_image=image,state=state,timed_parameters=raw,prediction=pred)
        record=dict(reason='ok',kind=FORMAT,capture_stamp=cloud['stamp'],capture_age_s=q['t']-cloud['stamp'],
            utility=utility.tolist(),direct_plans=plans,preprocess_ms=(prepared-begin)*1000,inference_ms=(inferred-prepared)*1000)
        save(folder/f'{index:05d}.json',record);record['frontend_ms']=(time.perf_counter()-begin)*1000
        return plans,record
