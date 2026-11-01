"""Transport-free PX4 setpoint adapter. No arming, device access or flight node.

The future ROS 2 node must run this sampler independently of planning, publish
OffboardControlMode and TrajectorySetpoint at 50 Hz, and use its ROS clock in
microseconds (with configured uXRCE-DDS synchronization). The estimator-frame
alignment must be calibrated, not inferred from a camera optical frame.
"""
import math
import numpy as np
from navigation_common import Curve

def wrap(x):return (x+math.pi)%(2*math.pi)-math.pi

class FrameAlignment:
    def __init__(self,px4_origin_in_world,world_to_enu_yaw):
        self.origin=np.asarray(px4_origin_in_world,float);c,s=math.cos(world_to_enu_yaw),math.sin(world_to_enu_yaw)
        if self.origin.shape!=(3,) or not np.isfinite(self.origin).all() or not math.isfinite(world_to_enu_yaw):raise ValueError('invalid measured frame alignment')
        self.rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        self.yaw=world_to_enu_yaw
    def ned(self,value,position=False):
        p=self.rotation@(np.asarray(value)-(self.origin if position else 0));return [float(p[1]),float(p[0]),float(-p[2])]

def messages(state,yaw,yaw_rate,stamp_us,alignment):
    if not isinstance(stamp_us,int) or stamp_us<0:raise ValueError('timestamp must be integer ROS clock microseconds')
    if not np.isfinite([*state['p'],*state['v'],*state['a'],yaw,yaw_rate]).all():raise ValueError('nonfinite setpoint')
    heartbeat=dict(timestamp=stamp_us,position=True,velocity=False,acceleration=False,attitude=False,body_rate=False,thrust_and_torque=False,direct_actuator=False)
    setpoint=dict(timestamp=stamp_us,position=alignment.ned(state['p'],True),velocity=alignment.ned(state['v']),acceleration=alignment.ned(state['a']),yaw=wrap(math.pi/2-yaw-alignment.yaw),yawspeed=-yaw_rate)
    return heartbeat,setpoint

class CommittedSchedule:
    """Only an already admitted nominal/backup pair may enter the executor.

    Call revoke() when a map guard fails. The supervisor must then select a
    separately verified escape or PX4 failsafe; this object cannot invent one.
    A map version is provenance, not a promise of future obstacle visibility.
    """
    def __init__(self,plans,checked_at,valid_until,map_version):
        if not plans or not checked_at<=valid_until:raise ValueError('invalid validity interval')
        self.curves=tuple(Curve(p) for p in plans);self.valid_until=valid_until;self.checked_at=checked_at;self.map_version=map_version;self.revoked=False
        for a,b in zip(self.curves,self.curves[1:]):
            if b.start<=a.start or b.start>a.start+a.duration+1e-8:raise ValueError('schedule gap or order')
            for k,tol in [('p',1e-6),('v',1e-6),('a',1e-5)]:
                if np.linalg.norm(np.asarray(a.state(b.start)[k])-b.state(b.start)[k])>tol:raise ValueError('discontinuous committed schedule')
        last=self.curves[-1]
        if np.linalg.norm(last.v(last.duration))>1e-6 or np.linalg.norm(last.a(last.duration))>1e-5:raise ValueError('cannot freeze a moving terminal command')
    def revoke(self):self.revoked=True
    def sample(self,t):
        if self.revoked or t>self.valid_until or t<self.checked_at or t<self.curves[0].start:raise RuntimeError('no current validated schedule')
        curve=self.curves[0]
        for c in self.curves[1:]:
            if c.start<=t:curve=c
            else:break
        return curve.state(t),curve.plan.get('kind','nominal')
