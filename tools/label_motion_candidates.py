"""Offline reevaluation of actual candidate curves and multi-time stopping probes."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from navigation_common import Worker,Curve,save
from motion_frontend import net_class,cloud_risk,SPEC
from direct_frontend import decode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);a=ap.parse_args()
    torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False);d=np.load(a.dataset/'dataset.npz');rows=json.loads((a.dataset/'records.json').read_text());replays=json.loads((a.dataset/'replays.json').read_text())
    images=d['image'];states=d['state']  # Decompress each training array once, not once per candidate bank.
    models={};labels={};manifest={}
    for bank in ['base','features']:
        for seed in (a.seeds if bank=='features' else [11]):
            path=(a.base if bank=='base' else a.features)/f'geometry_s{seed}.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
            channels=3 if bank=='base' else 6;model=net_class(channels)().eval();model.load_state_dict(ck['state_dict']);key=f'{bank}{seed}';models[key]=(model,channels)
            labels[key]=dict(parameters=np.zeros((len(rows),15,15),np.float32),quality=np.full((len(rows),15),-2.,np.float32),feasible=np.zeros((len(rows),15),np.float32),
                risk=np.ones((len(rows),15),np.float32),recovery=np.zeros((len(rows),15),np.float32),min_margin=np.full((len(rows),15),-100.,np.float32))
            manifest[key]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),channels=channels)
    details=[];seen=set();begin=time.perf_counter()
    for case,replay in replays.items():
        with Worker(replay['backend'],a.out/'workers'/case) as w:
            for event in replay['sequence']:
                if 'sample' not in event:w.request(event);continue
                i=event['sample'];seen.add(i);row=rows[i];q=row['query'];cloud=dict(row['cloud']);cloud['points']=np.fromfile(cloud['binary_xyz_f32'],dtype='<f4').reshape(-1,3)
                for key,(model,channels) in models.items():
                    with torch.inference_mode():raw,pred=model(torch.from_numpy(images[i:i+1,:channels]),torch.from_numpy(states[i:i+1]))
                    raw,pred=raw.numpy()[0],pred.numpy()[0];plans,_=decode(q,raw,pred);label=labels[key];label['parameters'][i]=raw
                    r=w.request(dict(q,op='evaluate_direct',direct_plans=plans,forbid_optimizer=True));assert r['optimizer_calls']==0
                    passed=[dict(id=b['id'],plan=b['plan']) for b in r['branches'] if b['ok']]
                    stops=w.request(dict(q,op='evaluate_recovery',direct_plans=passed,forbid_optimizer=True)) if passed else dict(branches=[])
                    stop_by_id={b['id']:b['stops'] for b in stops['branches']};diag=[]
                    for b in r['branches']:
                        j=b['id'];info=dict(id=j,reason=b['reason'])
                        if b['ok']:
                            c=Curve(b['plan']);goal=np.asarray(q['goal']);initial=np.linalg.norm(goal-q['start_state']['p'])
                            progress=(initial-np.linalg.norm(goal-c.s(min(.6,c.duration))))/.6;end=initial-np.linalg.norm(goal-c.s(c.duration))
                            length=np.linalg.norm(np.diff(c.s(np.linspace(0,c.duration,61)),axis=0),axis=1).sum()
                            risk=cloud_risk(b['plan'],cloud);recovery=np.mean([x['ok'] for x in stop_by_id[j]])
                            label['quality'][i,j]=progress+.12*end-.03*c.duration-.03*length;label['feasible'][i,j]=1
                            label['risk'][i,j]=risk['risk'];label['min_margin'][i,j]=risk['min_margin_m'];label['recovery'][i,j]=recovery
                            info.update(**risk,recovery=float(recovery),stops=stop_by_id[j])
                        diag.append(info)
                    details.append(dict(state=i,model=key,branches=diag))
        print(json.dumps(dict(case=case,states=len(seen),seconds=time.perf_counter()-begin)),flush=True)
    assert len(seen)==len(rows)
    for key,label in labels.items():np.savez_compressed(a.out/(key+'.npz'),**label)
    save(a.out/'details.json',details);save(a.out/'protocol.json',dict(states=len(rows),curves=len(rows)*15*len(models),models=manifest,spec=SPEC,
        risk='observed latest-cloud hit points; directional one-dimensional braking heuristic, not occupancy/free-space proof',
        recovery='six independently validated stop branches on the current fused map; not future-map guarantee or long-horizon RL return'))

if __name__=='__main__':main()
