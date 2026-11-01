import argparse,json,random
from pathlib import Path
from collections import Counter
import numpy as np
from navigation_common import ROOT,make_scene,save
from backup_trial import trial,conditions
from perception_trial import percentiles
from d435i_adapter import install_simulator,CAMERA,camera_scene as make_scene
from motion_frontend import MotionFrontend
from run_motion_experiment import freeze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['pilot','test','stress'],required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    policies=['repair','control11','features11','labels11','both11','both22','both33'] if a.stage=='test' else ['repair','control11','both11']
    scenes=[(make_scene('empty'),'development'),(make_scene('forest',3202),'validation')] if a.stage=='pilot' else [(make_scene('forest',s),'camera_test') for s in [4281,4282,4283,4284]] if a.stage=='test' else [(make_scene('forest',3202),'range3'),(make_scene('forest',3202),'holes30'),(make_scene('forest',3202),'late650')]
    jobs=[dict(scene=s,split=split,backend=b,policy=p) for s,split in scenes for b in ['ego','rog'] for p in policies];random.Random(202610315).shuffle(jobs);freeze(a.out,jobs,new_test_seeds=[4281,4282,4283,4284]);save(a.out/'camera.json',CAMERA)
    models={(p,b):None if p=='repair' else MotionFrontend(b,ROOT/'motion_models'/f'{p[:-2]}_s{p[-2:]}.pt') for p in policies for b in ['ego','rog']};results=[]
    for job in jobs:
        s=job['scene'];p=job['policy'];b=job['backend'];split=job['split'];name=f"{s['name']}_{s['seed']}_{b}_{p}_{split}";folder=a.out/name
        install_simulator(max_depth=3. if split=='range3' else None,holes=.3 if split=='holes30' else 0.)
        kwargs=dict(timeout_after=5.) if split=='late650' else {}
        r=trial(b,s,conditions()[0],12000,folder,frontend=models[p,b],plan_budget_ms=100.,**kwargs)
        timing=json.loads((folder/'timings.json').read_text());requests=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()];ps=[x for x in timing if x['op']=='plan'];tr=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1)
        r.update(folder=name,policy=p,seed=s['seed'],split=split,plan_requests=len(ps),budget_rejections=sum(x.get('plan_budget_exceeded',False) for x in ps),
            optimizer_calls=sum(x['response']['optimizer_calls'] for x in requests),plan_rpc_ms=percentiles([x['rpc_ms'] for x in ps]),
            path_length_m=float(np.linalg.norm(np.diff(tr[:,1:4],axis=0),axis=1).sum()),
            frontend_ms=percentiles([x['neural_input']['frontend_ms'] for x in ps if 'neural_input' in x]),
            inference_ms=percentiles([x['neural_input']['inference_ms'] for x in ps if 'inference_ms' in x.get('neural_input',{})]),
            direct_rejections=dict(Counter(c['reason'] for x in ps for c in x.get('direct_checks',[]) if c['reason']!='ok')))
        if p!='repair':assert r['optimizer_calls']==0
        save(folder/'result.json',r);results.append(r);save(a.out/'summary.json',results);print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','plan_rpc_ms']}),flush=True)

if __name__=='__main__':main()
