#!/usr/bin/env python3
"""Interleave routing policies with identical scenes, limits and execution rules."""
import argparse,itertools,json
from pathlib import Path
from navigation_common import CONFIG,make_scene,save
from run_navigation_trials import trial

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--policies',nargs='+',choices=['full','reuse','repair'],default=['full','reuse','repair'])
    p.add_argument('--backends',nargs='+',choices=['ego','rog'],default=['ego','rog'])
    p.add_argument('--scenes',nargs='+',default=['offset_wall','forest']);p.add_argument('--seeds',type=int,nargs='+',default=[11])
    p.add_argument('--delays',type=float,nargs='+',default=[0.]);p.add_argument('--repeat',type=int,default=1);p.add_argument('--max-time',type=float,default=35.)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'summary.json').exists():raise FileExistsError(a.out)
    save(a.out/'experiment.json',dict(policies=a.policies,backends=a.backends,scenes=a.scenes,seeds=a.seeds,delays=a.delays,repeat=a.repeat,common_config=CONFIG))
    results=[];case=0
    for scene in a.scenes:
        for seed,delay,backend,repeat in itertools.product(a.seeds if scene=='forest' else [0],a.delays,a.backends,range(a.repeat)):
            shift=case%len(a.policies);policies=a.policies[shift:]+a.policies[:shift];case+=1
            for policy in policies:
                name=f'{scene}_{seed}_{backend}_{policy}_delay{round(delay*1000)}_r{repeat}'
                target=a.out/name
                if target.exists():raise FileExistsError(target)
                r=trial(backend,make_scene(scene,seed),delay,target,a.max_time,policy);r['repeat']=repeat;r['folder']=name
                results.append(r);save(a.out/'summary.json',results)
                print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','final_error_m','full_astar_calls','local_astar_calls','reused_routes','repaired_routes','rejections']}),flush=True)
if __name__=='__main__':main()
