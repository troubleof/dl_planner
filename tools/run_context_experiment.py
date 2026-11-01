import argparse,hashlib,json,random
from pathlib import Path
from backup_trial import *
from context_frontend import ContextFrontend
from adapted_frontends import RangeFrontend

SPLITS=dict(train=list(range(3101,3107)),validation=[3201,3202],test=[3401,3402,3403,3404])
def model(policy):
    if policy=='repair':return None
    if policy=='old11':return RangeFrontend(ROOT/'models/range_scorer_s11.pt')
    if policy=='context_repair':return ContextFrontend(mode='repair')
    if policy=='rule':return ContextFrontend(mode='rule')
    if policy=='ensemble':return ContextFrontend([ROOT/'context_models/round1'/f'multistep_s{s}.pt' for s in [11,22,33]])
    stage,target,seed=policy.split('_');return ContextFrontend([ROOT/'context_models'/stage/f'{target}_s{seed}.pt'])

def freeze(out,jobs):
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.relative_to(ROOT).parts[0] in ['src','tools','config','third_party','models','context_models'] and '__pycache__' not in str(p)]
    save(out/'frozen_manifest.json',dict(source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},jobs=jobs,splits=SPLITS,
        protocol='5Hz x 12240 rays; same EGO/P0 executor; CPU and context overhead measured; 80ms extra delay; 2s branch labels with continuation recorded per dataset; finite candidate family'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['collect','aggregate','pilot','test'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--policies',nargs='+');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    if a.stage=='collect':spec=[(make_scene('forest',s),split) for split in ['train','validation'] for s in SPLITS[split]];policies=['context_repair']
    elif a.stage=='aggregate':spec=[(make_scene('forest',s),'train') for s in SPLITS['train']];policies=['round0_multistep_11']
    elif a.stage=='pilot':spec=[(make_scene('empty'),'development'),(make_scene('forest',3202),'validation')];policies=['repair','context_repair','rule']
    else:spec=[(make_scene('forest',s),'test') for s in SPLITS['test']]+[(make_scene('forest',3301),'regression'),(make_scene('offset_wall'),'regression')];policies=['repair','old11','context_repair','rule','round0_short_11','round0_multistep_11','round1_multistep_11','round1_multistep_22','round1_multistep_33','ensemble']
    policies=a.policies or policies;frontends={p:model(p) for p in policies};save(a.out/'models.json',{p:m.metadata for p,m in frontends.items() if m is not None})
    jobs=[dict(backend=b,scene=s,policy=p,split=split) for b in ['ego','rog'] for s,split in spec for p in policies];random.Random(202610301).shuffle(jobs);freeze(a.out,jobs);results=[]
    for job in jobs:
        scene=job['scene'];name=f"{scene['name']}_{scene['seed']}_{job['backend']}_{job['policy']}";folder=a.out/name
        r=trial(job['backend'],scene,conditions()[0],12000,folder,frontend=frontends[job['policy']])
        timing=json.loads((folder/'timings.json').read_text());req=[x for x in timing if x['op']=='plan'];plans=json.loads((folder/'plans.json').read_text());traj=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1)
        r.update(folder=name,policy=job['policy'],seed=scene['seed'],split=job['split'],repeat=0,path_length_m=float(np.linalg.norm(np.diff(traj[:,1:4],axis=0),axis=1).sum()),
            proposal_commits=sum(p['nominal'].get('frontend')=='yopo' for p in plans),plan_rpc_ms=percentiles([x['rpc_ms'] for x in req]),
            inference_ms=percentiles([x['neural_input']['inference_ms'] for x in req if 'inference_ms' in x.get('neural_input',{})]),
            context_rpc_ms=percentiles([x['context_rpc_ms'] for x in req if 'context_rpc_ms' in x]),all_rpc_cpu_s=sum(x['rpc_ms'] for x in timing)/1000)
        save(folder/'result.json',r);results.append(r);save(a.out/'summary.json',results)
        print(json.dumps({k:r[k] for k in ['folder','outcome','duration_s','proposal_commits','plan_rpc_ms','context_rpc_ms']}),flush=True)
if __name__=='__main__':main()
