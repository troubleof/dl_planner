"""Small frozen P0 matrix; failure cases are never dropped or reclassified."""
import argparse,hashlib
from backup_trial import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True,help='Frozen perception-matrix-v1 directory (or extracted original_failures)')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for backend in ['ego','rog']:
        jobs.append(dict(name=f'{backend}_offset_delay400',backend=backend,scene=make_scene('offset_wall'),condition='delay_400'))
        jobs.append(dict(name=f'{backend}_historical_prefix_branch',backend=backend,scene=make_scene('offset_wall'),condition='delay_400',
            replay_prefix=str(a.baseline/f'offset_wall_0_{backend}_delay_400_r0')))
    for backend in ['ego','rog']:
        for sc in [make_scene('wall_gap'),make_scene('offset_wall')]+[make_scene('forest',s) for s in [41,42,101,102]]:
            jobs.append(dict(name=f"{backend}_{sc['name']}_{sc['seed']}_baseline",backend=backend,scene=sc,condition='baseline'))
        jobs.append(dict(name=f'{backend}_compute_timeout',backend=backend,scene=make_scene('empty'),condition='baseline',timeout_after=3.,max_time=9.))
        jobs.append(dict(name=f'{backend}_new_obstacle',backend=backend,scene=make_scene('empty'),condition='baseline',
            obstacle_event=dict(t=3.,obstacle=dict(kind='box',lo=[-3.,-.3,0.],hi=[-2.7,.3,5.]))))
        jobs.append(dict(name=f'{backend}_late_obstacle',backend=backend,scene=make_scene('empty'),condition='baseline',
            obstacle_event=dict(t=3.,obstacle=dict(kind='box',lo=[-4.9,-2.,0.],hi=[-4.6,2.,5.]))))
    tracked=[p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] in ['src','tools','config'] and '__pycache__' not in str(p)]
    save(a.out/'frozen_manifest.json',dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},jobs=jobs,
        notes=['P0 nominal plus precommitted stop schedule. 0.6 s branch offset, finite 11-duration family.',
               'Serialized map guard / solve / validate; full remainder checked at each RPC, latest arrival commit gate unchanged.',
               'Original sampled centre/free and inflated obstacle checks retained. No whole swept-volume certificate.',
               'Timeout adds 650 ms solve delay from t=3; not a separate real-time safety thread.',
               'New/late obstacles are synthetic step changes to test response, not a moving-obstacle model.']))
    rows=[]
    for job in jobs:
        options={k:v for k,v in job.items() if k in ['timeout_after','max_time','obstacle_event','replay_prefix']}
        r=trial(job['backend'],job['scene'],next(c for c in conditions() if c['name']==job['condition']),12000,a.out/job['name'],**options)
        r['folder']=job['name'];rows.append(r);save(a.out/'summary.json',rows)
        print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','accepted_plans','backup_activations','final_speed']}),flush=True)

if __name__=='__main__':main()
