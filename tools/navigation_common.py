"""Shared sensor, process transport, and ideal B-spline executor. No leader inputs."""
import hashlib,json,math,os,select,subprocess,time
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'config/navigation.json').read_text())
BODY_DIMS=np.asarray(CONFIG['body_dimensions_m']);BODY_RADIUS=float(np.linalg.norm(BODY_DIMS)/2)
RADIUS=BODY_RADIUS+CONFIG['safety_margin_m']
CONFIG.update(body_radius_m=BODY_RADIUS,planning_radius_m=RADIUS,inflation_steps=math.ceil(RADIUS/CONFIG['resolution']))

def save(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class Worker:
    def __init__(self,backend,out,route_policy=None,frontend=None):
        self.frontend=frontend;self.received_cloud=None;self.backend=backend
        if frontend is not None:frontend.worker=self
        if frontend is not None and hasattr(frontend,'reset'):frontend.reset()
        self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.stdout=(self.out/'worker.stdout.log').open('wb');self.stderr=(self.out/'worker.stderr.log').open('wb');self.buffer=b''
        self.requests=(self.out/'requests.jsonl').open('w');self.request_id=0
        (self.out/'clouds').mkdir(exist_ok=True)
        env=os.environ.copy();env.update(ROS_DOMAIN_ID='89',ROS_LOCALHOST_ONLY='1',ROS_LOG_DIR=str(self.out/'ros_logs'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        lo,hi=np.asarray(CONFIG['bounds']);rog=dict(resolution=CONFIG['resolution'],inflation_resolution=CONFIG['resolution'],inflation_step=CONFIG['inflation_steps'],unk_inflation_en=False,
            map_size=(hi-lo).tolist(),fix_map_origin=((hi+lo)/2).tolist(),map_sliding=dict(enable=False,threshold=1.),
            virtual_ground_height=float(lo[2]),virtual_ceil_height=float(hi[2]),point_filt_num=1,intensity_thresh=-1.,load_pcd_en=False,frontier_extraction_en=False,
            ros_callback=dict(enable=False),visualization=dict(enable=False,range=(hi-lo).tolist()),esdf=dict(enable=False),
            raycasting=dict(enable=True,batch_update_size=1,unk_thresh=1.,ray_range=[0.,CONFIG['sensor_range']],local_update_box=(2*(hi-lo)).tolist(),**{k:CONFIG[k] for k in ['p_hit','p_miss','p_min','p_max','p_occ','p_free']}))
        save(self.out/'rog_runtime.json',dict(rog_map=rog))
        effective_config=dict(CONFIG)
        effective_config['track_observation_age']=bool(getattr(frontend,'requires_context',False))
        if route_policy is not None:effective_config['route_policy']=route_policy
        save(self.out/'navigation_runtime.json',effective_config)
        cmd=[str(ROOT/'install/pointcloud_nav/lib/pointcloud_nav/pointcloud_nav_worker'),backend,str(self.out/'rog_runtime.json'),str(self.out/'navigation_runtime.json')]
        self.p=subprocess.Popen(cmd,cwd=self.out,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,env=env)
        save(self.out/'command.json',cmd)
        self._read(b'NAV_READY',timeout=45.)
    def _read(self,prefix,timeout=45.):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1);self.stdout.write(line+b'\n');self.stdout.flush()
                if line.startswith(prefix):return line[len(prefix):].strip()
                continue
            if self.p.poll() is not None:raise RuntimeError(f'worker exited {self.p.returncode}: {self.out}')
            ready,_,_=select.select([self.p.stdout],[],[],max(0.,end-time.monotonic()))
            if ready:
                chunk=os.read(self.p.stdout.fileno(),65536)
                if not chunk:raise RuntimeError(f'worker output closed: {self.out}')
                self.buffer+=chunk
        raise TimeoutError(f'worker timeout: {self.out}')
    def request(self,obj):
        begin=time.perf_counter();obj=dict(obj);context_result=None
        if obj.get('cloud') and 'points' in obj['cloud']:
            self.received_cloud=obj['cloud']
            if self.frontend is not None and hasattr(self.frontend,'observe'):self.frontend.observe(obj['cloud'])
        enabled=self.frontend is not None and obj['op']=='plan' and not obj.get('_skip_frontend',False)
        if enabled and (getattr(self.frontend,'requires_context',False) or getattr(self.frontend,'visible_context',False)):
            context_request={k:v for k,v in obj.items() if k in ['t','start_state','goal','cloud','execution_schedule','nominal_plan']}
            context_result=self.request(dict(context_request,op='visible_context' if getattr(self.frontend,'visible_context',False) else 'context'))
            obj.pop('cloud',None);obj['map_context']=context_result.get('context');obj['frontend_backend']=self.backend
            obj['context_ok']=context_result['ok'];obj['context_rpc_ms']=context_result['rpc_ms']
        self.request_id+=1
        neural=None
        if enabled:
            routes,neural=self.frontend.propose(self.received_cloud,obj,self.out/'neural',self.request_id)
            if getattr(self.frontend,'direct_trajectory',False):
                obj['direct_plans']=routes
                obj['direct_fallback_repair']=self.frontend.fallback_repair
                obj['forbid_optimizer']=not self.frontend.fallback_repair
            else:obj['neural_routes']=routes
            obj['neural_input']=neural
        if obj.get('cloud') and 'points' in obj['cloud']:
            c=dict(obj['cloud']);xyz=np.asarray(c.pop('points'),dtype='<f4').reshape(-1,3)
            path=self.out/'clouds'/f'{self.request_id:05d}.bin';xyz.tofile(path)
            c.update(binary_xyz_f32=str(path),count=len(xyz));obj['cloud']=c
            if c.get('sensor')=='d435i' and 'depth_grid_world' in c:
                depth_path=self.out/'clouds'/f'{self.request_id:05d}.depth.npz'
                np.savez(depth_path,world=np.asarray(c.pop('depth_grid_world')),valid=np.asarray(c.pop('depth_valid'),dtype=np.uint8))
                c['depth_evidence_npz']=str(depth_path)
        wire=json.dumps({k:v for k,v in obj.items() if k!='map_context'},separators=(',',':'),allow_nan=False).encode()+b'\n'
        budget=getattr(self,'plan_budget_ms',getattr(self.frontend,'plan_budget_ms',None)) if obj['op']=='plan' else None
        if budget is not None:
            obj['worker_budget_ms']=max(0.,budget-(time.perf_counter()-begin)*1000)
            wire=json.dumps({k:v for k,v in obj.items() if k!='map_context'},separators=(',',':'),allow_nan=False).encode()+b'\n'
        self.p.stdin.write(wire);self.p.stdin.flush()
        result=json.loads(self._read(b'NAV_RESPONSE '));result['rpc_ms']=(time.perf_counter()-begin)*1000;result['wire_bytes']=len(wire)
        if context_result is not None:
            result['context_rpc_ms']=context_result['rpc_ms'];result['map_update_ms']+=context_result['map_update_ms']
            result['observation_update_ms']=context_result.get('observation_update_ms',0.)
        if neural is not None:result['neural_input']=neural
        if obj['op']=='plan' and result.get('ok') and 'plan' in result:result['plan']['frontend']=result.get('frontend','repair')
        if budget is not None:
            result['plan_budget_ms']=budget
            result['plan_budget_exceeded']=result['rpc_ms']>budget
            if result['plan_budget_exceeded']:
                result['before_budget_ok']=result['ok'];result['ok']=False;result['reason']='plan_compute_budget'
        self.requests.write(json.dumps(dict(request=obj,response=result),allow_nan=False)+'\n');self.requests.flush()
        # Research evidence I/O is also charged to simulated execution time.
        result['rpc_before_evidence_ms']=result['rpc_ms'];result['rpc_ms']=(time.perf_counter()-begin)*1000
        if budget is not None and result['rpc_ms']>budget:
            result['plan_budget_exceeded']=True;result['ok']=False;result['reason']='plan_compute_budget'
        if 'error' in result:raise ValueError(result['error'])
        return result
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:self.p.wait(timeout=5.)
            except subprocess.TimeoutExpired:self.p.terminate();self.p.wait(timeout=5.)
        self.stdout.close();self.stderr.close();self.requests.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

class Curve:
    def __init__(self,plan):
        self.plan=plan;self.start=plan['start'];self.duration=plan['duration']
        q=np.asarray(plan['controls']);dt=plan['dt'];knots=(np.arange(len(q)+4)-3)*dt
        if abs(self.duration-knots[-4])>1e-8:raise ValueError('inconsistent spline duration')
        self.duration=float(knots[-4])
        self.s=BSpline(knots,q,3,extrapolate=False);self.v=self.s.derivative();self.a=self.v.derivative()
    def state(self,t):
        u=np.clip(t-self.start,0.,self.duration)
        # Curves end at rest; never freeze a moving command on a missed replan.
        p=self.s(u);v=self.v(u);a=self.a(u)
        if t>=self.start+self.duration:v=np.zeros(3);a=np.zeros(3)
        return dict(p=p.tolist(),v=v.tolist(),a=a.tolist())

def stationary(p):return dict(p=list(p),v=[0.,0.,0.],a=[0.,0.,0.])

def make_scene(name,seed=0):
    obstacles=[]
    if name=='wall_gap':
        obstacles=[dict(kind='box',lo=[-.25,-5.8,0.],hi=[.25,-.9,5.]),dict(kind='box',lo=[-.25,.9,0.],hi=[.25,5.8,5.])]
    elif name=='offset_wall':
        obstacles=[dict(kind='box',lo=[-.25,-5.8,0.],hi=[.25,.4,5.])]
    elif name=='forest':
        rng=np.random.default_rng(seed)
        for _ in range(18):
            x,y=rng.uniform([-4.5,-4.5],[4.5,4.5]);r=float(rng.uniform(.18,.35))
            if not any(o['kind']=='cylinder' and np.linalg.norm(np.array([x,y])-o['xy'])<r+o['radius']+.9 for o in obstacles):
                obstacles.append(dict(kind='cylinder',xy=[float(x),float(y)],radius=r,z=[0.,5.]))
    elif name!='empty':raise ValueError(name)
    return dict(name=name,seed=seed,start=[-7.,0.,2.5],goal=[7.,0.,2.5],obstacles=obstacles)

class Sensor:
    def __init__(self,azimuths=None):
        azimuths=azimuths or CONFIG['sensor_azimuths']
        az=np.arange(azimuths)*2*np.pi/azimuths
        # Dense, limited vertical LiDAR fan; this is a horizontal-navigation baseline.
        low,high,step=CONFIG['sensor_elevations_degrees'];el=np.deg2rad(np.arange(low,high+step/2,step));a,e=np.meshgrid(az,el)
        self.d=np.column_stack([(np.cos(e)*np.cos(a)).ravel(),(np.cos(e)*np.sin(a)).ravel(),np.sin(e).ravel()])
    def capture(self,scene,p,stamp):
        p=np.asarray(p);d=self.d;sensor_range=CONFIG['sensor_range'];t=np.full(len(d),sensor_range+.1)
        for o in scene['obstacles']:
            if o['kind']=='cylinder':
                q=p[:2]-o['xy'];A=np.sum(d[:,:2]**2,axis=1);B=d[:,:2]@q;C=q@q-o['radius']**2
                disc=B*B-A*C;near=(-B-np.sqrt(np.maximum(disc,0)))/A
                z=p[2]+near*d[:,2];valid=(disc>=0)&(near>=0)&(z>=o['z'][0])&(z<=o['z'][1])
            else:
                lo=np.asarray(o['lo']);hi=np.asarray(o['hi']);inv=np.divide(1.,d,out=np.full_like(d,np.inf),where=np.abs(d)>1e-12)
                t0=(lo-p)*inv;t1=(hi-p)*inv
                near=np.minimum(t0,t1).max(axis=1);far=np.maximum(t0,t1).min(axis=1);valid=(near<=far)&(near>=0)
            t=np.where(valid&(near<=sensor_range)&(near<t),near,t)
        points=(p+d*t[:,None]).astype(np.float32).astype(np.float64)
        return dict(frame='world',stamp=float(stamp),origin=p.tolist(),points=points.tolist())

def clearance(scene,points):
    p=np.asarray(points);out=np.full(p.shape[:-1],100.)
    for o in scene['obstacles']:
        if o['kind']=='cylinder':
            radial=np.linalg.norm(p[...,:2]-o['xy'],axis=-1)-o['radius']
            vertical=np.maximum(o['z'][0]-p[...,2],p[...,2]-o['z'][1])
            dist=np.linalg.norm(np.stack([np.maximum(radial,0),np.maximum(vertical,0)],axis=-1),axis=-1)+np.minimum(np.maximum(radial,vertical),0)
        else:
            lo=np.asarray(o['lo']);hi=np.asarray(o['hi']);q=np.abs(p-(lo+hi)/2)-(hi-lo)/2
            dist=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(axis=-1),0)
        out=np.minimum(out,dist)
    return out

def round_tick(t):return math.ceil((t-1e-10)/CONFIG['dt'])*CONFIG['dt']

class CommitGate:
    """A result cannot be backdated, retimed, or committed after a map change."""
    @staticmethod
    def admit(plan,publish_t,checked_version,current_version,expected,actual):
        if publish_t>=plan['start']-CONFIG['deadline_guard']-1e-10:return False,'deadline'
        if checked_version!=current_version:return False,'map_changed'
        if any(np.linalg.norm(np.asarray(expected[k])-actual[k])>tol for k,tol in [('p',1e-6),('v',1e-6),('a',1e-5)]):return False,'handoff_state_changed'
        return True,'ok'
