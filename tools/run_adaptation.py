"""Predeclared train/validation/new-map test splits in the existing simulator."""
import argparse,hashlib,json,random
from backup_trial import *
from yopo_frontend import YopoFrontend
from adapted_frontends import AlignedYopo,RangeFrontend

SPLITS=dict(train=list(range(3101,3107)),validation=[3201,3202],test=[3301,3302,3303,3304])

def freeze(out,jobs):
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] in ['src','tools','config','third_party','models'] and '__pycache__' not in str(p)]
    save(out/'frozen_manifest.json',dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},jobs=jobs,splits=SPLITS,
        protocol='Same P0 executor, 5 Hz x 12240, 0 transport delay, measured online frontend time + 80 ms fixed compute delay; no full-map model input.'))

def run_job(job,out,frontend):
    scene=job['scene'];name=f"{scene['name']}_{scene['seed']}_{job['backend']}_{job['policy']}"
    r=trial(job['backend'],scene,conditions()[0],12000,out/name,frontend=frontend)
    plans=json.loads((out/name/'plans.json').read_text());timings=json.loads((out/name/'timings.json').read_text());req=[x for x in timings if x['op']=='plan']
    trajectory=np.loadtxt(out/name/'trajectory.csv',delimiter=',',skiprows=1)
    r.update(folder=name,policy=job['policy'],seed=scene['seed'],split=job['split'],repeat=0,
        path_length_m=float(np.linalg.norm(np.diff(trajectory[:,1:4],axis=0),axis=1).sum()),
        proposal_commits=sum(p['nominal'].get('frontend')=='yopo' for p in plans),
        plan_requests=len(req),plan_rpc_ms=percentiles([x['rpc_ms'] for x in req]),
        inference_ms=percentiles([x['neural_input']['inference_ms'] for x in req if 'inference_ms' in x.get('neural_input',{})]),
        frontend_ms=percentiles([x['neural_input']['frontend_ms'] for x in req if 'neural_input' in x]),
        full_astar_calls=sum(x.get('full_astar_calls',0) for x in req),local_astar_calls=sum(x.get('local_astar_calls',0) for x in req))
    save(out/name/'result.json',r)
    print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','proposal_commits','accepted_plans','plan_rpc_ms']}),flush=True)
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['collect','pilot','test'],required=True)
    ap.add_argument('--policies',nargs='+');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if a.stage=='collect':spec=[(make_scene('forest',s),k) for k in ['train','validation'] for s in SPLITS[k]];policies=['repair']
    elif a.stage=='pilot':spec=[(make_scene('empty'),'development'),(make_scene('forest',3202),'validation')];policies=['repair','aligned','rule','learn11']
    else:spec=[(make_scene('forest',s),'test') for s in SPLITS['test']]+[(make_scene('offset_wall'),'regression'),(make_scene('forest',211),'regression')];policies=['repair','yopo','aligned','rule','learn11','learn22','learn33']
    policies=a.policies or policies;models={'repair':None}
    for p in policies:
        if p=='repair':continue
        if p=='yopo':models[p]=YopoFrontend()
        elif p=='aligned':models[p]=AlignedYopo()
        elif p=='rule':models[p]=RangeFrontend()
        else:models[p]=RangeFrontend(ROOT/'models'/f'range_scorer_s{p[5:]}.pt')
    save(a.out/'models.json',{p:m.metadata for p,m in models.items() if m is not None})
    jobs=[dict(backend=b,scene=s,policy=p,split=split) for b in ['ego','rog'] for s,split in spec for p in policies]
    random.Random(20261030).shuffle(jobs);freeze(a.out,jobs);results=[]
    for job in jobs:results.append(run_job(job,a.out,models[job['policy']]));save(a.out/'summary.json',results)

if __name__=='__main__':main()
