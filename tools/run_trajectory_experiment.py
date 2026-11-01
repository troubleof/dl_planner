"""Predeclared fresh maps; full failures retained; unchanged closed-loop trial."""
import argparse
import hashlib
import random
from pathlib import Path
from backup_trial import *
from run_adaptation import run_job
from trajectory_frontend import TrajectoryFrontend
from adapted_frontends import RangeFrontend

POLICIES=['repair','old11','fixed11','trajectory11','trajectory22','trajectory33']
SPLITS=dict(train=list(range(3101,3107)),validation=[3201,3202],test=[3501,3502,3503,3504])


def frontend(policy):
    if policy=='repair':return None
    if policy=='old11':return RangeFrontend(ROOT/'models/range_scorer_s11.pt')
    fixed=policy.startswith('fixed')
    seed=policy.replace('fixed','').replace('trajectory','')
    return TrajectoryFrontend(ROOT/'trajectory_models'/f'trajectory_s{seed}.pt',fixed=fixed)


def freeze(out,jobs):
    hashes={}
    for base in ['src','tools','config','third_party','models','trajectory_models']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(out/'frozen_manifest.json',dict(source_sha256=hashes,jobs=jobs,splits=SPLITS,
        protocol='Unchanged P0 executor, 5Hz/12240 rays, measured CPU time plus 80ms; 35s failure cap; no test-based selection',
        ablation='fixed11 shares exact weights/scores with trajectory11 but sets geometry residual to zero; old11 is historical scorer',
        primary='All three trajectory seeds vs paired repair; all failures counted as 35s; report backend/new-map/regression separately'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','test'],required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--policies',nargs='+',choices=POLICIES)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    policies=a.policies or (['repair','fixed11','trajectory11'] if a.stage=='pilot' else POLICIES)
    scenes=([(make_scene('empty'),'development'),(make_scene('forest',3202),'validation')] if a.stage=='pilot' else
            [(make_scene('forest',s),'test') for s in SPLITS['test']]+[(make_scene('offset_wall'),'regression')])
    models={p:frontend(p) for p in policies}
    save(a.out/'models.json',{p:m.metadata for p,m in models.items() if m is not None})
    jobs=[dict(backend=b,scene=s,policy=p,split=split) for b in ['ego','rog'] for s,split in scenes for p in policies]
    random.Random(20261031).shuffle(jobs);freeze(a.out,jobs);results=[]
    for job in jobs:
        r=run_job(job,a.out,models[job['policy']])
        timings=json.loads((a.out/r['folder']/'timings.json').read_text())
        r['all_rpc_wall_s']=sum(x['rpc_ms'] for x in timings)/1000
        save(a.out/r['folder']/'result.json',r)
        results.append(r);save(a.out/'summary.json',results)


if __name__=='__main__':main()
