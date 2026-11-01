"""Reevaluate exact model outputs and six independent stop probes, offline."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from navigation_common import Worker,Curve,save
from visible_frontend import network_class
from motion_frontend import cloud_risk
from direct_frontend import decode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--geometry',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    data=np.load(a.dataset/'dataset.npz');arrays={k:data[k] for k in ['image','state','seed','features']};rows=json.loads((a.dataset/'records.json').read_text());replays=json.loads((a.dataset/'replays.json').read_text());models={};labels={};manifest={};details=[];seen=set();begin=time.perf_counter()
    for seed in a.seeds:
        p=a.geometry/f'geometry_s{seed}.pt';ck=torch.load(p,map_location='cpu',weights_only=True);model=network_class()().eval();model.load_state_dict(ck['state_dict']);models[seed]=model;manifest[seed]=dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest());labels[seed]=dict(parameters=np.zeros((len(rows),15,15),np.float32),quality=np.full((len(rows),15),-2.,np.float32),feasible=np.zeros((len(rows),15),np.float32),risk=np.ones((len(rows),15),np.float32),recovery=np.zeros((len(rows),15),np.float32))
    for case,replay in replays.items():
        with Worker(replay['backend'],a.out/'workers'/case) as w:
            for event in replay['sequence']:
                if 'sample' not in event:w.request(event);continue
                i=event['sample'];seen.add(i);row=rows[i];q=row['query'];cloud=dict(row['cloud']);cloud['points']=np.fromfile(cloud['binary_xyz_f32'],dtype='<f4').reshape(-1,3)
                for seed,model in models.items():
                    with torch.inference_mode():raw,pred=model(*(torch.from_numpy(arrays[k][i:i+1]) for k in ['image','state','seed','features']))
                    raw,pred=raw.numpy()[0],pred.numpy()[0];plans,_=decode(q,raw,pred);lab=labels[seed];lab['parameters'][i]=raw
                    result=w.request(dict(q,op='evaluate_direct',direct_plans=plans,forbid_optimizer=True));assert result['optimizer_calls']==0;passed=[dict(id=b['id'],plan=b['plan']) for b in result['branches'] if b['ok']]
                    recovery=w.request(dict(q,op='evaluate_recovery',direct_plans=passed,forbid_optimizer=True)) if passed else dict(branches=[]);stop_by_id={b['id']:b['stops'] for b in recovery['branches']};diag=[]
                    for b in result['branches']:
                        j=b['id'];info=dict(id=j,reason=b['reason'])
                        if b['ok']:
                            c=Curve(b['plan']);goal=np.asarray(q.get('local_progress_target',q['goal']));initial=np.linalg.norm(goal-q['start_state']['p']);progress=(initial-np.linalg.norm(goal-c.s(min(.6,c.duration))))/.6;end=initial-np.linalg.norm(goal-c.s(c.duration));risk=cloud_risk(b['plan'],cloud);recovery_value=np.mean([x['ok'] for x in stop_by_id[j]])
                            lab['quality'][i,j]=progress+.12*end-.03*c.duration;lab['feasible'][i,j]=1;lab['risk'][i,j]=risk['risk'];lab['recovery'][i,j]=recovery_value;info.update(**risk,recovery=float(recovery_value),stops=stop_by_id[j])
                        diag.append(info)
                    details.append(dict(state=i,seed=seed,branches=diag))
        print(json.dumps(dict(case=case,states=len(seen),seconds=time.perf_counter()-begin)),flush=True)
    assert len(seen)==len(rows)
    for seed,lab in labels.items():np.savez_compressed(a.out/f's{seed}.npz',**lab)
    save(a.out/'details.json',details);save(a.out/'protocol.json',dict(states=len(rows),models=manifest,branches=len(rows)*15*len(models),recovery='Six finite stopping branches on current received map; not a guarantee for future maps',risk='Observed hit-point directional braking margin; never used as free-space proof'))

if __name__=='__main__':main()
