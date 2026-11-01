import argparse,json,random,hashlib
from pathlib import Path
from collections import Counter
from navigation_common import ROOT,save
from backup_trial import trial,conditions
from d435i_adapter import camera_scene
from camera_yaw import CameraYaw,install
from visible_frontend import VisibleFrontend
from motion_frontend import MotionFrontend
from perception_trial import percentiles

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['pilot','collect','onpolicy','test','regression','stress'],required=True);ap.add_argument('--policies',nargs='+',default=['repair','short_fixed','short_scan']);ap.add_argument('--weights',type=Path,default=ROOT/'visible_models');ap.add_argument('--test-seeds',nargs='+',type=int,default=[4381,4382,4383,4384]);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if a.stage=='pilot':scenes=[(camera_scene('empty'),'development'),(camera_scene('forest',3202),'validation')]
    elif a.stage=='collect':scenes=[(camera_scene('forest',s),'train') for s in range(3101,3107)]+[(camera_scene('empty'),'train'),(camera_scene('forest',3201),'validation'),(camera_scene('forest',3202),'validation')]
    elif a.stage=='onpolicy':scenes=[(camera_scene('forest',s),'train') for s in [3101,3102]]
    elif a.stage=='test':scenes=[(camera_scene('forest',s),'test') for s in a.test_seeds]
    elif a.stage=='regression':scenes=[(camera_scene('forest',s),'seen_regression') for s in [4281,4282,4283,4284]]
    else:scenes=[(camera_scene('forest',3202),s) for s in ['range3','holes30','late650','budget25']]
    jobs=[dict(scene=s,split=split,backend=b,policy=p) for s,split in scenes for b in ['ego','rog'] for p in a.policies];random.Random(202610316).shuffle(jobs)
    hashes={}
    for d in ['src','tools','config','motion_models','visible_models']:
        for f in (ROOT/d).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts:hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    save(a.out/'frozen_manifest.json',dict(source_sha256=hashes,jobs=jobs,test_seeds=a.test_seeds,yaw_rate_limit=.7853981633974483))
    results=[]
    for job in jobs:
        s,b,p,split=job['scene'],job['backend'],job['policy'],job['split'];enabled=p not in ['repair','old','short_fixed'];yaw=CameraYaw(enabled)
        install(yaw,max_depth=3. if split=='range3' else None,holes=.3 if split=='holes30' else 0.)
        if p.startswith('repair'):model=None
        elif p.startswith('old'):model=MotionFrontend(b,ROOT/'motion_models/both_s11.pt')
        elif p.startswith('learn') or p.startswith('hybrid'):model=VisibleFrontend(b,a.weights/f'visible_s{p[-2:]}.pt',hybrid=p.startswith('hybrid'))
        else:model=VisibleFrontend(b)
        name=f"{s['name']}_{s['seed']}_{b}_{p}_{split}";folder=a.out/name
        r=trial(b,s,conditions()[0],12000,folder,frontend=model,camera_yaw=yaw,plan_budget_ms=25. if split=='budget25' else 100.,timeout_after=5. if split=='late650' else None)
        timing=json.loads((folder/'timings.json').read_text());requests=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()];ps=[x for x in timing if x['op']=='plan']
        r.update(folder=name,policy=p,seed=s['seed'],split=split,optimizer_calls=sum(x['response']['optimizer_calls'] for x in requests),plan_requests=len(ps),
            budget_rejections=sum(x.get('plan_budget_exceeded',False) for x in ps),plan_rpc_ms=percentiles([x['rpc_ms'] for x in ps]),
            frontend_ms=percentiles([x['neural_input']['frontend_ms'] for x in ps if 'neural_input' in x]),inference_ms=percentiles([x['neural_input']['inference_ms'] for x in ps if 'inference_ms' in x.get('neural_input',{})]),
            fixed_fallback_commits=sum(x['response'].get('ok',False) and x['response'].get('direct_selected_id',0)>=15 for x in requests),
            direct_rejections=dict(Counter(c['reason'] for x in ps for c in x.get('direct_checks',[]) if c['reason']!='ok')))
        if not p.startswith('repair'):assert r['optimizer_calls']==0
        save(folder/'result.json',r);results.append(r);save(a.out/'summary.json',results)
        print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','accepted_plans','final_error_m','plan_rpc_ms']}),flush=True)

if __name__=='__main__':main()
