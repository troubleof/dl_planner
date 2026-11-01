"""Ideal bounded yaw actuator for visibility experiments. No instantaneous look-at."""
import math
import numpy as np

def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi

class CameraYaw:
    def __init__(self,enabled=True):
        self.enabled=enabled;self.yaw=0.;self.rate=0.;self.last=None;self.stalled=0.;self.records=[]
    def update(self,t,state,goal):
        dt=0. if self.last is None else t-self.last;self.last=t
        velocity=np.asarray(state['v']);speed=np.linalg.norm(velocity[:2]);heading=math.atan2(goal[1]-state['p'][1],goal[0]-state['p'][0])
        if speed<.08:self.stalled+=dt
        else:self.stalled=0.
        target=heading
        if self.stalled>1.:
            # Alternating side observations while stationary. No hidden map query.
            phase=(self.stalled-1.)%12
            offset=math.radians(110)*(1 if phase<6 else -1)
            target=heading+offset
        elif speed>.15:target=math.atan2(velocity[1],velocity[0])
        if self.enabled and dt>0:
            desired=np.clip(2*wrap(target-self.yaw),-math.pi/4,math.pi/4)
            self.rate+=float(np.clip(desired-self.rate,-math.pi/2*dt,math.pi/2*dt));self.yaw=wrap(self.yaw+self.rate*dt)
        self.records.append([t,self.yaw,self.rate,target])
    def rotation(self):
        from d435i_adapter import OPTICAL_TO_WORLD
        c,s=np.cos(self.yaw),np.sin(self.yaw);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])@OPTICAL_TO_WORLD

def install(yaw,max_depth=None,holes=0.):
    import backup_trial,perception_trial
    from d435i_adapter import D435iSensor,from_depth
    from navigation_common import Sensor,CONFIG
    class Camera(D435iSensor):
        def capture(self,scene,p,stamp):
            R=yaw.rotation();rays=self.optical.reshape(-1,3)@R.T;self.d=rays/np.linalg.norm(rays,axis=1)[:,None]
            raw=Sensor.capture(self,scene,p,stamp);delta=np.asarray(raw['points'])-np.asarray(p)
            radial=np.linalg.norm(delta,axis=1);depth=(delta@R)[:,2].reshape(self.optical.shape[:2]);depth[radial.reshape(depth.shape)>CONFIG['sensor_range']]=0.
            if holes:depth[np.random.default_rng(int(round(stamp*1000))+scene['seed']).random(depth.shape)<holes]=0.
            return from_depth(depth,intrinsics=self.intrinsics,depth_scale=1.,camera_to_world=R,origin=p,stamp=stamp,pose_stamp=stamp,clock_domain='navigation',block=1,max_depth=max_depth)
    def factory(azimuths=None):return Camera(max_depth=max_depth,holes=holes)
    backup_trial.Sensor=factory;perception_trial.Sensor=factory
