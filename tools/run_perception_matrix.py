#!/usr/bin/env python3
"""Predeclared, interleaved one-factor perception stress tests. Never overwrite."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from perception_trial import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    cases=conditions();scenes=[make_scene('wall_gap'),make_scene('offset_wall')]+[make_scene('forest',s) for s in [41,42,101,102]]
    if a.pilot:
        cases=[c for c in cases if c['name'] in ['baseline','rate_frame_20','points_1530','delay_400','drop_50']]
        scenes=[make_scene('forest',11)]
    jobs=[]
    for scene in scenes:
        for backend in ['ego','rog']:
            for c in cases:
                jobs.append(dict(scene=scene,backend=backend,condition=c,repeat=0,cohort='primary'))
            if not a.pilot and scene['name']=='forest':
                for c in cases:
                    if c['family'] in ['independent_drop','burst_drop']:
                        for repeat in [1,2]:jobs.append(dict(scene=scene,backend=backend,condition=c,repeat=repeat,cohort='loss_repeats'))
    # Fixed shuffle spreads conditions across wall-clock thermal/load changes.
    rng=np.random.default_rng(20261029);rng.shuffle(jobs)
    tracked=[p for p in ROOT.rglob('*') if p.is_file() and (str(p.relative_to(ROOT)).startswith(('src/pointcloud_nav/','tools/','config/')) or p.name in ['build_navigation.sh','run_navigation.sh']) and '__pycache__' not in str(p)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    manifest=dict(source_sha256=hashes,jobs=jobs,expected_runs=len(jobs),fixed_extra_compute_delay_s=.08,max_time_s=35.,
        common_boot='Four unmodified dense scans at t=0,.2,.4,.6; impairments begin at t=1.',
        notes=['C++ planner and map code unchanged. repair policy fixed.','Snapshot ray model: 17 vertical rays, no measurement noise or intra-scan motion.','Latest received scan replaces older unconsumed scans; no extra mapper thread.','Sensor transport delay is separate from fixed 80 ms planning delay.'])
    f=a.out/'frozen_manifest.json'
    if f.exists():
        old=json.loads(f.read_text());assert old==manifest,'source or experiment changed'
        if not a.resume:raise FileExistsError(a.out)
    else:save(f,manifest)
    rows=[]
    for i,job in enumerate(jobs):
        sc=job['scene'];c=job['condition'];backend=job['backend'];repeat=job['repeat']
        folder=f"{sc['name']}_{sc['seed']}_{backend}_{c['name']}_r{repeat}"
        target=a.out/folder
        if target.exists():
            if not a.resume or not (target/'result.json').exists():raise FileExistsError(target)
            r=json.loads((target/'result.json').read_text())
        else:
            seed=10000+sc['seed']*31+(['wall_gap','offset_wall','forest'].index(sc['name']))*1000+repeat*100000
            r=trial(backend,sc,c,seed,target)
        r.update(folder=folder,repeat=repeat,cohort=job['cohort']);rows.append(r);save(a.out/'summary.json',rows)
        print(json.dumps(dict(progress=f'{i+1}/{len(jobs)}',folder=folder,outcome=r['outcome'],flight_s=round(r['duration_s'],2),accepted=r['accepted_plans'],rejections=r['rejections'])),flush=True)
    print('Completed',len(rows),'trials',flush=True)
if __name__=='__main__':main()
