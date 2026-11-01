#!/usr/bin/env python3
"""Matched-input routing comparison; do not interpret replay as closed-loop success."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from navigation_common import Worker,ROOT,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source-root',type=Path,default=ROOT/'runs/navigation-benchmark-v1');p.add_argument('--seeds',nargs='+',type=int,default=[41,42,43]);p.add_argument('--repeat',type=int,default=2);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);summary=[]
    if (a.out/'summary.json').exists():raise FileExistsError(a.out)
    for backend in ['ego','rog']:
        for seed in a.seeds:
            source=a.source_root/f'forest_{seed}_{backend}_delay0/worker/requests.jsonl'
            requests=[json.loads(line)['request'] for line in source.read_text().splitlines()]
            payload_hash=hashlib.sha256(json.dumps(requests,sort_keys=True).encode()).hexdigest()
            clouds={req['cloud']['binary_xyz_f32']:hashlib.sha256(Path(req['cloud']['binary_xyz_f32']).read_bytes()).hexdigest() for req in requests if 'cloud' in req}
            for repeat in range(a.repeat):
                order=['full','reuse','repair'];shift=(repeat+seed)%3;order=order[shift:]+order[:shift]
                for policy in order:
                    target=a.out/f'{backend}_forest{seed}_{policy}_r{repeat}'
                    if target.exists():raise FileExistsError(target)
                    values=[]
                    with Worker(backend,target,route_policy=policy) as w:
                        for req in requests:
                            r=w.request(req)
                            if req['op']=='plan':values.append(r)
                    def pct(k):
                        v=[x[k] for x in values if k in x];return dict(zip(['p50','p95','p99'],map(float,np.percentile(v,[50,95,99])))) if v else {}
                    row=dict(backend=backend,seed=seed,policy=policy,repeat=repeat,requests_sha256=payload_hash,cloud_sha256=clouds,
                        planning_requests=len(values),valid_candidates=sum(x['ok'] for x in values),full_astar_calls=sum(x.get('full_astar_calls',0) for x in values),local_astar_calls=sum(x.get('local_astar_calls',0) for x in values),
                        reused_routes=sum(x.get('route_source')=='reuse' for x in values),repaired_routes=sum(x.get('route_source')=='repair' for x in values),search_ms=pct('search_ms'),rpc_ms=pct('rpc_ms'))
                    summary.append(row);save(a.out/'summary.json',summary)
                    print(json.dumps({k:v for k,v in row.items() if k!='cloud_sha256'}),flush=True)
if __name__=='__main__':main()
