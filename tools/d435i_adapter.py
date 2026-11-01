"""D435i rectified depth contract, bounded downsampling, and ideal camera simulator.

No device is claimed connected. Caller supplies calibrated intrinsics and a
capture-time optical-camera pose in the same clock/world frame as navigation.
Zero/out-of-range depth is unknown, never a free-space/no-return ray.
"""
import json,time
import numpy as np
from navigation_common import ROOT,CONFIG,Sensor

CAMERA=json.loads((ROOT/'config/d435i.json').read_text())
OPTICAL_TO_WORLD=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])

def camera_scene(name,seed=0):
    from navigation_common import make_scene
    scene=make_scene(name,seed)
    # Explicit room surfaces supply physical depth returns. They are present for
    # every policy, teacher rendering and truth evaluator; no max-range free ray.
    scene['obstacles']+= [dict(kind='box',lo=lo,hi=hi) for lo,hi in [
        ([-9.4,-5.6,0.],[9.4,5.6,.05]),([-9.4,-5.6,4.95],[9.4,5.6,5.]),
        ([9.2,-5.6,0.],[9.4,5.6,5.]),([-9.4,-5.6,0.],[-9.2,5.6,5.]),
        ([-9.4,-5.6,0.],[9.4,-5.4,5.]),([-9.4,5.4,0.],[9.4,5.6,5.])]]
    scene['camera_enclosure']=True
    return scene

def from_depth(depth,*,intrinsics,depth_scale,camera_to_world,origin,stamp,pose_stamp,clock_domain,block=8,max_depth=None):
    begin=time.perf_counter();depth=np.asarray(depth);R=np.asarray(camera_to_world,dtype=float);origin=np.asarray(origin,dtype=float)
    if depth.ndim!=2 or depth.shape[0]%block or depth.shape[1]%block:raise ValueError('image dimensions must divide block')
    if R.shape!=(3,3) or not np.isfinite(R).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or np.linalg.det(R)<.999:raise ValueError('invalid camera rotation')
    if origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('invalid camera origin')
    if not np.isfinite([stamp,pose_stamp,depth_scale]).all() or depth_scale<=0 or abs(stamp-pose_stamp)>1e-6:raise ValueError('capture-time interpolated pose required')
    if clock_domain!='navigation':raise ValueError('convert camera hardware timestamp into navigation clock before use')
    if intrinsics.get('distortion','none')!='none':raise ValueError('provide rectified depth/intrinsics or SDK deprojected rays')
    fx,fy,cx,cy=[float(intrinsics[k]) for k in ['fx','fy','cx','cy']]
    if not np.isfinite([fx,fy,cx,cy]).all() or min(fx,fy)<=0:raise ValueError('invalid intrinsics')
    z=depth.astype(np.float32)*depth_scale;h,w=z.shape;maximum=CAMERA['maximum_depth_m'] if max_depth is None else max_depth
    valid=np.isfinite(z)&(z>=CAMERA['minimum_depth_m'])&(z<=maximum)
    # Pick the nearest actual valid pixel in each block, and keep ITS ray.
    grid=np.where(valid,z,np.inf).reshape(h//block,block,w//block,block).transpose(0,2,1,3).reshape(h//block,w//block,-1)
    ids=np.argmin(grid,axis=-1);selected=np.take_along_axis(grid,ids[...,None],axis=-1)[...,0];ok=np.isfinite(selected)
    rows=np.arange(h//block)[:,None]*block+ids//block;cols=np.arange(w//block)[None,:]*block+ids%block
    selected=np.where(ok,selected,0);optical=np.stack([(cols-cx)*selected/fx,(rows-cy)*selected/fy,selected],axis=-1)
    radial=np.linalg.norm(optical,axis=-1);ok &= radial<=CONFIG['sensor_range']
    world=optical@R.T+origin;world=np.where(ok[...,None],world,origin)
    cloud=dict(frame='world',stamp=float(stamp),origin=origin.tolist(),points=world[ok].tolist(),sensor='d435i',
        camera_to_world=R.tolist(),depth_shape=list(ok.shape),depth_grid_world=world.tolist(),depth_valid=ok.astype(np.uint8).tolist(),
        intrinsics=dict(intrinsics),depth_scale=depth_scale,pose_stamp=pose_stamp,clock_domain=clock_domain,
        projection_ms=(time.perf_counter()-begin)*1000)
    return cloud

class D435iSensor(Sensor):
    def __init__(self,azimuths=None,max_depth=None,holes=0.):
        w,h=CAMERA['simulation_grid'];fx=w/(2*np.tan(np.deg2rad(CAMERA['simulation_hfov_degrees']/2)));fy=h/(2*np.tan(np.deg2rad(CAMERA['simulation_vfov_degrees']/2)))
        self.intrinsics=dict(fx=fx,fy=fy,cx=(w-1)/2,cy=(h-1)/2,distortion='none');u,v=np.meshgrid(np.arange(w),np.arange(h))
        self.optical=np.stack([(u-(w-1)/2)/fx,(v-(h-1)/2)/fy,np.ones_like(u)],axis=-1)
        ray=self.optical.reshape(-1,3)@OPTICAL_TO_WORLD.T;self.d=ray/np.linalg.norm(ray,axis=1)[:,None]
        self.max_depth=max_depth;self.holes=holes
    def capture(self,scene,p,stamp):
        raw=super().capture(scene,p,stamp);points=np.asarray(raw['points']);delta=points-np.asarray(p)
        distance=np.linalg.norm(delta,axis=1);depth=(delta@OPTICAL_TO_WORLD)[:,2].reshape(self.optical.shape[:2]);depth[distance.reshape(depth.shape)>CONFIG['sensor_range']]=0.
        if self.holes:
            rng=np.random.default_rng(int(round(stamp*1000))+scene['seed']);depth[rng.random(depth.shape)<self.holes]=0
        return from_depth(depth,intrinsics=self.intrinsics,depth_scale=1.,camera_to_world=OPTICAL_TO_WORLD,origin=p,
                          stamp=stamp,pose_stamp=stamp,clock_domain='navigation',block=1,max_depth=self.max_depth)

def install_simulator(max_depth=None,holes=0.):
    # Existing simulator and executor are reused; only sensor construction changes.
    import backup_trial,perception_trial
    def factory(azimuths=None):return D435iSensor(max_depth=max_depth,holes=holes)
    backup_trial.Sensor=factory;perception_trial.Sensor=factory

def camera_observation(cloud,q,backend,channels=6):
    from motion_frontend import directional,SPEC
    from adapted_frontends import rotation
    p=np.asarray(cloud['origin']);world=np.asarray(cloud['depth_grid_world']);valid=np.asarray(cloud['depth_valid'],bool)
    if world.shape!=(*valid.shape,3) or not np.isfinite(world).all():raise ValueError('invalid depth grid')
    R=np.asarray(cloud['camera_to_world']);flu=np.column_stack([R[:,2],-R[:,0],-R[:,1]])
    ranges=np.linalg.norm(world-p,axis=-1);base=np.stack([np.where(valid,ranges/CONFIG['sensor_range'],0),valid,valid]).astype(np.float32)
    cap=cloud.get('capture_state',dict(v=[0,0,0],a=[0,0,0]));s=q['start_state'];goal=np.asarray(q['goal'])-s['p'];distance=np.linalg.norm(goal);age=q['t']-cloud['stamp']
    state=np.r_[np.asarray(cap['v'])@flu/3,np.asarray(cap['a'])@flu/6,np.asarray(s['v'])@flu/3,np.asarray(s['a'])@flu/6,
                (np.asarray(s['p'])-p)@flu/3,goal@flu/max(distance,1e-6),distance/20,age/.5,(age+CONFIG['handoff'])/.8,(CONFIG['handoff']-CONFIG['deadline_guard']-.08)/.3]
    rot,_=rotation(q['goal'],np.asarray(s['p']));state=np.r_[state,float(backend=='rog'),np.asarray(s['v'])@rot/3,np.asarray(s['a'])@rot/6,goal@rot/20].astype(np.float32)
    if channels==3:return base,state
    closing,acceleration,margin=directional(world,s['p'],s['v'],s['a']);extra=np.stack([closing/3,acceleration/6,np.clip(margin/SPEC['margin_scale_m'],-1,1)]).astype(np.float32)*valid[None]
    return np.concatenate([base,extra]),state
