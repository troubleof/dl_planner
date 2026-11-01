import argparse,hashlib,json,random
from pathlib import Path
from backup_trial import *
from run_adaptation import run_job
from recovery_frontend import RecoveryFrontend
from direct_frontend import DirectFrontend

POLICIES=['repair','recovery11','recovery33','direct11','direct22','direct33','direct_repair11']
def frontend(policy,backend):
    if policy=='repair':return None
    if policy.startswith('recovery'):return RecoveryFrontend(backend,ROOT/'recovery_models'/f'long_s{policy[-2:]}.pt')
    return DirectFrontend(backend,ROOT/'direct_models_v2'/f'direct_s{policy[-2:]}.pt',fallback_repair=policy.startswith('direct_repair'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['pilot','test','stress'],required=True)
    ap.add_argument('--policies',nargs='+',choices=POLICIES);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    policies=a.policies or (['repair','recovery11','direct11','direct_repair11'] if a.stage!='test' else POLICIES)
    if a.stage=='pilot':scenes=[(make_scene('empty'),'development'),(make_scene('forest',3202),'validation')]
    elif a.stage=='test':scenes=[(make_scene('forest',s),'test') for s in range(3701,3705)]+[(make_scene('forest',3503),'seen_regression'),(make_scene('offset_wall'),'seen_regression')]
    else:scenes=[(make_scene('forest',3202),'stress_timeout'),(make_scene('empty'),'stress_obstacle')]
    models={(p,b):frontend(p,b) for p in policies for b in ['ego','rog']}
    jobs=[dict(backend=b,scene=s,policy=p,split=split) for s,split in scenes for b in ['ego','rog'] for p in policies]
    random.Random(202610313).shuffle(jobs)
    hashes={}
    for base in ['src','tools','config','models','trajectory_models','recovery_models','direct_models_v2']:
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(a.out/'frozen_manifest.json',dict(source_sha256=hashes,jobs=jobs,new_test_seeds=list(range(3701,3705)),unchanged_executor=True,
        strict_direct='no online EGO; rejects retain committed stopping fallback',hybrid='direct_repair is separately counted optimizer fallback'))
    save(a.out/'models.json',{p+'_'+b:m.metadata for (p,b),m in models.items() if m is not None})
    results=[]
    for job in jobs:
        if a.stage!='stress':row=run_job(job,a.out,models[job['policy'],job['backend']])
        else:
            s=job['scene'];name=f"{s['name']}_{s['seed']}_{job['backend']}_{job['policy']}"
            options=dict(timeout_after=5.) if job['split']=='stress_timeout' else dict(obstacle_event=dict(t=5.,obstacle=dict(kind='box',lo=[-2.6,-.45,0.],hi=[-2.2,.45,5.])))
            row=trial(job['backend'],s,conditions()[0],12000,a.out/name,frontend=models[job['policy'],job['backend']],**options)
            row.update(folder=name,policy=job['policy'],seed=s['seed'],split=job['split'])
        folder=a.out/row['folder'];timing=json.loads((folder/'timings.json').read_text());plans=json.loads((folder/'plans.json').read_text());requests=[json.loads(l) for l in (folder/'worker/requests.jsonl').read_text().splitlines()]
        row.update(all_rpc_wall_s=sum(t['rpc_ms'] for t in timing)/1000,optimizer_calls=sum(r['response']['optimizer_calls'] for r in requests),
            direct_commits=sum(p['nominal'].get('frontend')=='direct' for p in plans),ego_commits=sum(p['nominal'].get('frontend')!='direct' for p in plans),
            optimizer_fallback_requests=sum(t.get('optimizer_fallback',False) for t in timing),
            direct_rejections=dict(Counter(c['reason'] for t in timing for c in t.get('direct_checks',[]) if c['reason']!='ok')))
        if job['policy'] in ['direct11','direct22','direct33']:assert row['optimizer_calls']==0 and row['ego_commits']==0
        save(folder/'result.json',row);results.append(row);save(a.out/'summary.json',results)

if __name__=='__main__':main()
