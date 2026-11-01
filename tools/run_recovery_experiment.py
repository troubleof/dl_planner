"""Same D435i simulator, maps, EGO baseline and validator; isolated outputs."""
import argparse,json,random,hashlib
from pathlib import Path
from collections import Counter
from navigation_common import ROOT,save
from backup_trial import trial,conditions
from d435i_adapter import camera_scene
from camera_yaw import CameraYaw,install
from visible_frontend import VisibleFrontend
from perception_trial import percentiles

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--split',choices=['train','validation','test','regression','stress'],required=True);ap.add_argument('--seeds',type=int,nargs='+',required=True);ap.add_argument('--policies',nargs='+',required=True);ap.add_argument('--weights',type=Path,default=ROOT/'survival_models');ap.add_argument('--max-time',type=float,default=35.);ap.add_argument('--condition',choices=['baseline','late650','budget25','holes30','range3'],default='baseline');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    jobs=[dict(seed=s,backend=b,policy=p) for s in a.seeds for b in ['ego','rog'] for p in a.policies];random.Random(202610317).shuffle(jobs)
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for d in ['src','tools','config','visible_models','survival_models'] for f in (ROOT/d).rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    save(a.out/'frozen_manifest.json',dict(source_sha256=hashes,jobs=jobs,split=a.split,simulator='D435i 106x60, 5 Hz, bounded yaw, ideal pose/translation'))
    results=[]
    for job in jobs:
        s,b,p=job['seed'],job['backend'],job['policy'];yaw=CameraYaw(True);install(yaw,max_depth=3. if a.condition=='range3' else None,holes=.3 if a.condition=='holes30' else 0.);scene=camera_scene('forest',s)
        recover=p.endswith('_recover') or p.startswith(('mlp','spatial','temporal','riskrule','bankrule'))
        if p.startswith('repair'):model=None
        elif p.startswith('old'):model=VisibleFrontend(b,ROOT/'visible_models/visible_s11.pt')
        elif p.startswith(('mlp','spatial','temporal','riskrule','bankrule')):
            from recovery_frontend import RecoveryFrontend
            model=RecoveryFrontend(b,None if p in ['riskrule','bankrule'] else a.weights/(p+'.pt'),plain_rule=p=='bankrule')
        else:model=VisibleFrontend(b)
        name=f'forest_{s}_{b}_{p}_{a.split}';folder=a.out/name
        r=trial(b,scene,conditions()[0],12000,folder,frontend=model,camera_yaw=yaw,plan_budget_ms=25. if a.condition=='budget25' else 100.,recovery_mode=recover,max_time=a.max_time,timeout_after=5. if a.condition=='late650' else None)
        timings=json.loads((folder/'timings.json').read_text());requests=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()];events=json.loads((folder/'events.json').read_text());ps=[x for x in timings if x['op']=='plan']
        r.update(folder=name,policy=p,seed=s,split=a.split,stress_condition=a.condition,recovery_mode=recover,optimizer_calls=sum(x['response']['optimizer_calls'] for x in requests),plan_requests=len(ps),plan_rpc_ms=percentiles([x['rpc_ms'] for x in ps]),escape_searches=sum(e['kind']=='escape_search' for e in events),escape_commits=sum(e['kind']=='escape_commit' for e in events),budget_rejections=sum(x.get('plan_budget_exceeded',False) for x in ps),inference_ms=percentiles([x['neural_input']['inference_ms'] for x in ps if 'inference_ms' in x.get('neural_input',{})]))
        if not p.startswith('repair'):assert r['optimizer_calls']==0
        save(folder/'result.json',r);results.append(r);save(a.out/'summary.json',results);print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','escape_searches','escape_commits','plan_rpc_ms']}),flush=True)

if __name__=='__main__':main()
