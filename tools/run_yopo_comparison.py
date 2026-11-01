"""Paired repair versus author-pretrained YOPO proposals + unchanged EGO backend."""
import argparse,hashlib,random
from backup_trial import *
from yopo_frontend import YopoFrontend

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--repeats',type=int,default=2);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);model=YopoFrontend();save(a.out/'model.json',model.metadata)
    scenes=[make_scene('empty'),make_scene('wall_gap'),make_scene('offset_wall')]+[make_scene('forest',s) for s in [41,42,101,102,211,212]]
    if a.pilot:scenes=[make_scene('offset_wall'),make_scene('forest',211)]
    jobs=[dict(backend=b,scene=s,policy=p,repeat=r) for b in ['ego','rog'] for s in scenes for p in ['repair','yopo'] for r in range(1 if a.pilot else a.repeats)]
    random.Random(20261029).shuffle(jobs)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] in ['src','tools','config','third_party'] and '__pycache__' not in str(p)]
    save(a.out/'frozen_manifest.json',dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},jobs=jobs,
        protocol='Existing P0 executor; baseline sensor 5Hz x 12240, zero transport delay, extra planning delay 80ms; CPU batch1; no training; newest scan and .5s age gates retained.'))
    results=[]
    for job in jobs:
        scene=job['scene'];name=f"{scene['name']}_{scene['seed']}_{job['backend']}_{job['policy']}_r{job['repeat']}"
        r=trial(job['backend'],scene,conditions()[0],12000,a.out/name,frontend=model if job['policy']=='yopo' else None)
        events=json.loads((a.out/name/'events.json').read_text());plans=json.loads((a.out/name/'plans.json').read_text())
        timings=json.loads((a.out/name/'timings.json').read_text());plan_requests=[x for x in timings if x['op']=='plan']
        trajectory=np.loadtxt(a.out/name/'trajectory.csv',delimiter=',',skiprows=1)
        r.update(folder=name,policy=job['policy'],repeat=job['repeat'],seed=scene['seed'],
            path_length_m=float(np.linalg.norm(np.diff(trajectory[:,1:4],axis=0),axis=1).sum()),
            yopo_commits=sum(p['nominal'].get('frontend')=='yopo' for p in plans),
            yopo_solved=sum(x.get('frontend')=='yopo' and x['ok'] for x in plan_requests),
            plan_requests=len(plan_requests),plan_rpc_ms=percentiles([x['rpc_ms'] for x in plan_requests]),
            inference_ms=percentiles([x['neural_input']['inference_ms'] for x in plan_requests if 'inference_ms' in x.get('neural_input',{})]),
            frontend_ms=percentiles([x['neural_input']['frontend_ms'] for x in plan_requests if 'neural_input' in x]),
            full_astar_calls=sum(x.get('full_astar_calls',0) for x in plan_requests),
            local_astar_calls=sum(x.get('local_astar_calls',0) for x in plan_requests))
        save(a.out/name/'result.json',r);results.append(r);save(a.out/'summary.json',results)
        print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','yopo_commits','accepted_plans','plan_rpc_ms']}),flush=True)

if __name__=='__main__':main()
