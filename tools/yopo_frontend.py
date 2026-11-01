"""Author YOPO-Simple weights as geometric proposals, never as executable commands.
Only already-delivered scans enter this module. Missing vertical FOV is zero depth.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from navigation_common import ROOT,CONFIG,save,stationary

class YopoFrontend:
    def __init__(self):
        begin=time.perf_counter()
        vendor=ROOT/'third_party/yopo';sys.path.insert(0,str(vendor/'YOPO'))
        import torch
        from config.config import cfg
        cfg['train']=False;cfg['velocity']=6.0 # preserve pretrained normalization
        from policy.yopo_network import YopoNetwork
        torch.set_num_threads(1)
        self.torch=torch;self.model=YopoNetwork().cpu().eval();self.transform=self.model.state_transform
        weight=vendor/'YOPO/saved/YOPO_1/epoch50.pth'
        self.model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
        self.H,self.W=cfg['image_height'],cfg['image_width'];self.duration=cfg['sgm_time']
        # Pixel centres in the author's forward/left/up camera convention.
        fx=(self.W/2)/np.tan(np.deg2rad(cfg['horizon_camera_fov']/2))
        fy=(self.H/2)/np.tan(np.deg2rad(cfg['vertical_camera_fov']/2))
        u,v=np.meshgrid(np.arange(self.W)+.5,np.arange(self.H)+.5)
        self.rays=np.stack([np.ones_like(u),(self.W/2-u)/fx,(self.H/2-v)/fy],axis=-1)
        self.rays/=np.linalg.norm(self.rays,axis=-1,keepdims=True)
        for _ in range(5):
            with torch.inference_mode():self.model(torch.zeros(1,1,self.H,self.W),torch.zeros(1,9,3,5))
        self.metadata=dict(upstream=json.loads((vendor/'upstream_manifest.json').read_text()),weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
            torch_version=torch.__version__,device='cpu',threads=1,load_warmup_ms=(time.perf_counter()-begin)*1000,
            image_shape=[self.H,self.W],missing_depth=0.,near_goal_repair_m=.5,top_k=3,training='none; unchanged author pretrained weights')

    def depth(self,cloud,yaw):
        xyz=np.asarray(cloud['points'],dtype=float);origin=np.asarray(cloud['origin']);delta=xyz-origin
        radii=np.linalg.norm(delta,axis=1)
        elevations=np.arange(-8.,8.01,1.);azimuths=len(delta)//17
        if azimuths*17!=len(delta):raise ValueError('adapter requires the existing organized 17-ring ray contract')
        az=(np.arctan2(self.rays[...,1],self.rays[...,0])+yaw)%(2*np.pi)
        el=np.rad2deg(np.arcsin(self.rays[...,2]));valid=(el>=-8)&(el<=8)
        ia=np.rint(az*azimuths/(2*np.pi)).astype(int)%azimuths
        ie=np.clip(np.rint(el+8).astype(int),0,16)
        # No-return endpoints encode free ray length only to the sensor range.
        r=np.minimum(radii.reshape(17,azimuths)[ie,ia],CONFIG['sensor_range'])
        depth=np.where(valid,r*self.rays[...,0],0.).astype(np.float32)/20.
        return depth,valid

    def propose(self,cloud,request,folder,index):
        begin=time.perf_counter();s=request['start_state'];goal=np.asarray(request['goal']);start=np.asarray(s['p'])
        if cloud is None:return [],dict(reason='no_received_cloud',frontend_ms=0.)
        if request['t']-cloud['stamp']>.5:return [],dict(reason='stale_received_cloud',frontend_ms=0.)
        if np.linalg.norm(goal-start)<.5:return [],dict(reason='near_goal_repair',frontend_ms=0.)
        capture=cloud.get('capture_state',stationary(cloud['origin']));origin=np.asarray(cloud['origin'])
        direction=goal-origin;yaw=np.arctan2(direction[1],direction[0]);c=np.cos(yaw);sine=np.sin(yaw)
        rot=np.array([[c,-sine,0],[sine,c,0],[0,0,1.]])
        depth,valid=self.depth(cloud,yaw)
        obs=np.concatenate([np.asarray(capture['v'])@rot,np.asarray(capture['a'])@rot,direction@rot])[None,:].astype(np.float32)
        normalized=self.transform.normalize_obs_cpu(obs);model_obs=self.transform.prepare_input_cpu(normalized)
        prepared=time.perf_counter()
        with self.torch.inference_mode():
            prediction,scores=self.model(self.torch.from_numpy(depth[None,None]),self.torch.from_numpy(model_obs))
        prediction=prediction.numpy().reshape(9,15).T;scores=scores.numpy().reshape(15)
        end=self.transform.pred_to_endstate_cpu(prediction,np.arange(14,-1,-1))
        inferred=time.perf_counter();routes=[]
        # Geometry only: fit a quintic with the author's predicted boundary
        # states. EGO independently assigns timing and future handoff P/V/A.
        for idx in np.argsort(scores):
            if not np.isfinite(end[idx]).all() or not np.isfinite(scores[idx]):continue
            p1=origin+end[idx,:3]@rot.T;v1=end[idx,3:6]@rot.T;a1=end[idx,6:9]@rot.T
            T=self.duration;p0=origin;v0=np.asarray(capture['v']);a0=np.asarray(capture['a'])
            M=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
            coef=np.linalg.solve(M,np.stack([p1-p0-v0*T-.5*a0*T*T,v1-v0-a0*T,a1-a0]))
            times=np.linspace(0,T,61);pts=p0+times[:,None]*v0+.5*times[:,None]**2*a0+np.stack([times**3,times**4,times**5],axis=1)@coef
            # Remove already-passed geometry, then explicitly join the predicted
            # handoff position. The join is checked by the existing map/optimizer.
            nearest=int(np.argmin(np.linalg.norm(pts-start,axis=1)))
            points=np.vstack([start,pts[nearest+1:]])
            if len(points)<2:continue
            routes.append(dict(id=int(idx),score=float(scores[idx]),points=points.tolist()))
            if len(routes)==3:break
        record=dict(reason='ok',capture_stamp=cloud['stamp'],capture_origin=cloud['origin'],capture_state=capture,
            handoff_state=request['start_state'],capture_age_s=request['t']-cloud['stamp'],yaw=yaw,observed_fraction=float(valid.mean()),
            preprocess_ms=(prepared-begin)*1000,inference_ms=(inferred-prepared)*1000,scores=scores.tolist(),decoded_endstates=end.tolist(),routes=routes)
        folder=Path(folder);folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{index:05d}.npz',depth=depth,observed_mask=valid,obs=model_obs,prediction=prediction)
        save(folder/f'{index:05d}.json',record)
        record['frontend_ms']=(time.perf_counter()-begin)*1000
        return routes,{k:record[k] for k in ['reason','capture_stamp','capture_age_s','observed_fraction','preprocess_ms','inference_ms','frontend_ms']}
