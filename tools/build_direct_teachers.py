"""Offline constrained spline projection. Teacher data never enters online observations."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from navigation_common import Curve,save
from direct_frontend import observation,PARAMS,K,BASE
from adapted_frontends import rotation

U=np.linspace(0,9,145)
B=BSpline((np.arange(16)-3),np.eye(12),3)(U)
FIT=np.linalg.pinv(B[:,3:9])

def fit(plan,q):
    c=Curve(plan);duration=c.duration
    if not .6<=duration<=5.95:return None,'duration'
    s=q['start_state'];p=np.asarray(s['p']);rot,_=rotation(q['goal'],p);v=np.asarray(s['v'])@rot;a=np.asarray(s['a'])@rot
    points=(c.s(U/9*duration)-p)@rot;h=duration/9
    controls=np.zeros((12,3));controls[:3]=[-v*h+a*h*h/3,-a*h*h/6,v*h+a*h*h/3];controls[9:]=points[-1]
    controls[3:9]=FIT@(points-B@controls)
    error=float(np.linalg.norm(B@controls-points,axis=1).max())
    if error>.12:return None,'projection_error'
    base=BASE@np.stack([controls[-1],v*duration,a*duration*duration])
    target=np.r_[np.log(duration),((controls[3:9,:2]-base[:,:2])/3).ravel(),controls[-1,:2]/3].astype(np.float32)
    return target,dict(projection_max_error_m=error,duration=duration)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'source_logs').mkdir();(a.out/'clouds').mkdir()
    images=[];states=[];targets=[];masks=[];records=[];replays={};hashes={};rejects={}
    old_records=json.loads((a.source/'runs/branch-labels-v1/records.json').read_text());case_meta={r['case']:r for r in old_records}
    sources=[(p.parent.name,p,False) for p in sorted((a.source/'runs/branch-labels-v1/branches').glob('*/requests.jsonl'))]
    sources += [(p.parent.parent.name+'_near',p,True) for p in sorted((a.source/'runs/train-rollouts-v1').glob('*/worker/requests.jsonl'))]
    for case,path,near in sources:
        sha=hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path)]=sha
        copied=a.out/'source_logs'/(case+'.jsonl');shutil.copy2(path,copied)
        sequence=[];latest=None;near_count=0
        if near:
            run=path.parents[1];meta=json.loads((run/'result.json').read_text());backend=meta['backend'];seed=meta['seed'];split=meta['split']
        else:meta=case_meta[case];backend=meta['backend'];seed=meta['seed'];split=meta['split']
        for entry_index,line in enumerate(path.read_text().splitlines()):
            entry=json.loads(line);q,r=entry['request'],entry['response']
            if 'cloud' in q:
                cloud=dict(q['cloud']);original=Path(cloud['binary_xyz_f32'])
                if not original.exists():original=path.parent/'clouds'/original.name
                cloud_name=case+'_'+str(entry_index)+'.bin';shutil.copy2(original,a.out/'clouds'/cloud_name)
                cloud['binary_xyz_f32']='clouds/'+cloud_name;latest=cloud
                sequence.append(dict(op='update',t=q['t'],cloud=cloud))
            if latest is None:continue
            if near:
                if q['op']!='plan' or not r.get('ok') or 'plan' not in r:continue
                distance=np.linalg.norm(np.asarray(q['goal'])-q['start_state']['p'])
                if distance>=3.:continue
                near_count+=1
                if near_count%3!=1 and distance>.4:continue
                teachers={i:r['plan'] for i in [2,7,12]}
            else:
                if q['op']!='evaluate_routes':continue
                teachers={i:b['plan'] for i,b in enumerate(r['branches'][:15]) if b['ok']}
            query={k:q[k] for k in ['t','start_state','goal']}
            cloud=dict(latest);cloud['points']=np.fromfile(a.out/cloud['binary_xyz_f32'],dtype='<f4').reshape(-1,3)
            image,state=observation(cloud,query,backend)
            target=np.zeros((K,PARAMS),np.float32);mask=np.zeros(K,np.float32);details={}
            for i,plan in teachers.items():
                projected,diagnostic=fit(plan,query)
                if projected is None:rejects[diagnostic]=rejects.get(diagnostic,0)+1;continue
                target[i]=projected;mask[i]=1;details[str(i)]=diagnostic
            index=len(records);images.append(image);states.append(state);targets.append(target);masks.append(mask)
            records.append(dict(index=index,case=case,backend=backend,seed=seed,split=split,near_goal=near,query=query,
                cloud=latest,source_sha256=sha,source_entry=entry_index,projection=details))
            sequence.append(dict(sample=index))
        replays[case]=dict(backend=backend,sequence=sequence)
    assert all(r['seed'] in (range(3101,3107) if r['split']=='train' else [3201,3202]) for r in records)
    np.savez_compressed(a.out/'dataset.npz',image=images,state=states,target=targets,mask=masks)
    save(a.out/'records.json',records);save(a.out/'replays.json',replays)
    save(a.out/'protocol.json',dict(states=len(records),train=sum(r['split']=='train' for r in records),validation=sum(r['split']=='validation' for r in records),
        near_goal_states=sum(r['near_goal'] for r in records),valid_teacher_curves=int(np.sum(masks)),projection_rejected=rejects,
        source_hashes=hashes,source=str(a.source),observations='only arrived cloud and current/handoff state; no teacher geometry in online input',
        teacher='offline EGO curves projected into 12-control cubic spline with fixed initial PVA and terminal rest',
        dataset_sha256=hashlib.sha256((a.out/'dataset.npz').read_bytes()).hexdigest()))
    print(json.dumps({k:v for k,v in json.loads((a.out/'protocol.json').read_text()).items() if k!='source_hashes'}),flush=True)

if __name__=='__main__':main()
