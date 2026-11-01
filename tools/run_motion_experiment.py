"""Frozen split, matched budgets and retained failures; online timing in executor."""
import argparse,hashlib,json,random
from collections import Counter
from pathlib import Path
from backup_trial import trial,conditions,make_scene,ROOT,save,np
from perception_trial import percentiles
from direct_frontend import DirectFrontend
from motion_frontend import MotionFrontend,SPEC

POLICIES=['repair','direct11','control11','features11','labels11','both11','both22','both33']
TEST_SEEDS=[4181,4182,4183,4184]

def frontend(policy,backend,budget=100.):
    if policy=='repair':return None
    if policy=='direct11':
        f=DirectFrontend(backend,ROOT/'direct_models_v2/direct_s11.pt');f.plan_budget_ms=budget;return f
    return MotionFrontend(backend,ROOT/'motion_models'/f'{policy[:-2]}_s{policy[-2:]}.pt',plan_budget_ms=budget)

def freeze(out,jobs,new_test_seeds=TEST_SEEDS):
    hashes={}
    for directory in ['src','tools','config','direct_models_v2','motion_models']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(out/'frozen_manifest.json',dict(source_sha256=hashes,jobs=jobs,new_test_seeds=new_test_seeds,
        protocol='5 Hz sensor, measured full Python/frontend/RPC/evidence time, additional fixed 80 ms; unchanged handoff/commit/map/stop gates',
        soft_budget='cooperative per-candidate check plus late-result rejection; no OS-level preemption/hard WCET guarantee'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','test','stress'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--policies',nargs='+',choices=POLICIES);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);jobs=[]
    policies=a.policies or (POLICIES if a.stage=='test' else ['repair','direct11','both11'] if a.stage=='pilot' else ['repair','both11'])
    if a.stage in ['pilot','test']:
        scenes=[(make_scene('empty'),'development'),(make_scene('forest',3202),'validation')] if a.stage=='pilot' else [(make_scene('forest',s),'test') for s in TEST_SEEDS]
        jobs=[dict(scene=s,split=split,policy=p,backend=b,budget=100.,options={}) for s,split in scenes for p in policies for b in ['ego','rog']]
    else:
        specs=[('budget25',make_scene('forest',3202),25.,{}),('budget50',make_scene('forest',3202),50.,{}),
               ('late650',make_scene('forest',3202),100.,dict(timeout_after=5.)),
               ('obstacle',make_scene('empty'),100.,dict(obstacle_event=dict(t=5.,obstacle=dict(kind='box',lo=[-2.6,-.45,0.],hi=[-2.2,.45,5.]))))]
        jobs=[dict(scene=s,split=label,policy=p,backend=b,budget=budget,options=opts) for label,s,budget,opts in specs for p in policies for b in ['ego','rog']]
    random.Random(202610314).shuffle(jobs);freeze(a.out,jobs)
    models={(p,b):frontend(p,b) for p in policies for b in ['ego','rog']};save(a.out/'models.json',{p+'_'+b:m.metadata for (p,b),m in models.items() if m})
    results=[]
    for job in jobs:
        s=job['scene'];p=job['policy'];b=job['backend'];name=f"{s['name']}_{s['seed']}_{b}_{p}_{job['split']}";folder=a.out/name
        model=models[p,b]
        if model:model.plan_budget_ms=job['budget']
        r=trial(b,s,conditions()[0],12000,folder,frontend=model,plan_budget_ms=job['budget'],**job['options'])
        timing=json.loads((folder/'timings.json').read_text());requests=[json.loads(v) for v in (folder/'worker/requests.jsonl').read_text().splitlines()];plans=json.loads((folder/'plans.json').read_text());tr=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1);ps=[t for t in timing if t['op']=='plan']
        events=json.loads((folder/'events.json').read_text());commits=[e for e in events if e['kind']=='commit']
        r.update(folder=name,policy=p,split=job['split'],seed=s['seed'],budget_ms=job['budget'],plan_requests=len(ps),
            optimizer_calls=sum(x['response']['optimizer_calls'] for x in requests),plan_rpc_ms=percentiles([x['rpc_ms'] for x in ps]),
            path_length_m=float(np.linalg.norm(np.diff(tr[:,1:4],axis=0),axis=1).sum()),budget_rejections=sum(x.get('plan_budget_exceeded',False) for x in ps),
            inference_ms=percentiles([x['neural_input']['inference_ms'] for x in ps if 'inference_ms' in x.get('neural_input',{})]),
            frontend_ms=percentiles([x['neural_input']['frontend_ms'] for x in ps if 'frontend_ms' in x.get('neural_input',{})]),
            mean_candidates_checked=float(np.mean([x.get('direct_tried',0) for x in ps])) if ps else 0.,
            direct_rejections=dict(Counter(c['reason'] for x in ps for c in x.get('direct_checks',[]) if c['reason']!='ok')),
            min_commit_slack_ms=min([(e['start']-.025-e['t'])*1000 for e in commits],default=None))
        if p!='repair':assert r['optimizer_calls']==0
        save(folder/'result.json',r);results.append(r);save(a.out/'summary.json',results)
        print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','plan_rpc_ms','budget_rejections']}),flush=True)

if __name__=='__main__':main()
