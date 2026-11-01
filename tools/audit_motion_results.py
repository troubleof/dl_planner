"""Independent model, admission, timing, PVA and executed-map replay audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from navigation_common import ROOT,Worker,Curve,CONFIG,save
from motion_frontend import net_class,observation
from direct_frontend import observation as old_observation,net_class as old_net,decode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);frozen=json.loads((a.run/'frozen_manifest.json').read_text());assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items())
    results=[];models={};maximum_jump=dict(p=0.,v=0.,a=0.);recomputed=0
    for row in json.loads((a.run/'summary.json').read_text()):
        folder=a.run/row['folder'];requests=[json.loads(l) for l in (folder/'worker/requests.jsonl').read_text().splitlines()];events=json.loads((folder/'events.json').read_text());plans=json.loads((folder/'plans.json').read_text());frames=json.loads((folder/'sensor_frames.json').read_text());traj=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1)
        latest=None;updates=[]
        for index,item in enumerate(requests,1):
            q,r=item['request'],item['response']
            if row['policy']!='repair':assert r['optimizer_calls']==0
            if 'cloud' in q:latest=q['cloud'];updates.append(q)
            if r.get('plan_budget_exceeded'):assert not r['ok']
            if q.get('neural_input',{}).get('reason')=='ok':
                ni=q['neural_input'];assert latest['stamp']==ni['capture_stamp']<=q['t']
                data=np.load(folder/'worker/neural'/f'{index:05d}.npz');spec=json.loads((folder/'model.json').read_text());key=row['policy']
                if key not in models:
                    path=Path(spec['weight']);assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256'];ck=torch.load(path,map_location='cpu',weights_only=True)
                    model=(old_net() if key=='direct11' else net_class(ck['channels']))().eval();model.load_state_dict(ck['state_dict']);models[key]=(model,ck.get('channels',3))
                model,channels=models[key];cloud=dict(latest);cloud['points']=np.fromfile(folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name,dtype='<f4').reshape(-1,3)
                if 'depth_evidence_npz' in cloud:
                    dep=np.load(folder/'worker/clouds'/Path(cloud['depth_evidence_npz']).name);cloud['depth_grid_world']=dep['world'];cloud['depth_valid']=dep['valid']
                im,st=old_observation(cloud,q,row['backend']) if key=='direct11' else observation(cloud,q,row['backend'],channels)
                assert np.array_equal(im,data['range_image']) and np.array_equal(st,data['state'])
                with torch.inference_mode():raw,pred=model(torch.from_numpy(im[None]),torch.from_numpy(st[None]))
                assert np.max(np.abs(raw.numpy()[0]-data['timed_parameters']))<1e-6;assert np.max(np.abs(pred.numpy()[0]-data['prediction']))<1e-6
                candidates,_=decode(q,raw.numpy()[0],pred.numpy()[0]);assert candidates==q['direct_plans'];recomputed+=1
            if r.get('ok') and r.get('frontend')=='direct':
                selected=next(c['plan'] for c in q['direct_plans'] if c['id']==r['direct_selected_id'])
                for k in ['start','dt','duration','controls']:assert np.allclose(selected[k],r['plan'][k],atol=1e-12,rtol=0)
        for pair in plans:
            for key in ['nominal','backup']:
                c=Curve(pair[key]);assert np.linalg.norm(c.v.c,axis=1).max()<=CONFIG['max_velocity']+1e-6;assert np.linalg.norm(c.a.c,axis=1).max()<=CONFIG['max_acceleration']+1e-6
                assert np.linalg.norm(c.v(c.duration))<1e-6 and np.linalg.norm(c.a(c.duration))<1e-5
            c,b=Curve(pair['nominal']),Curve(pair['backup'])
            for k,tol in [('p',1e-6),('v',1e-6),('a',1e-5)]:assert np.linalg.norm(np.asarray(c.state(b.start)[k])-b.state(b.start)[k])<tol
        for e in events:
            if e['kind']=='activate':
                for k,tol in [('p',1e-6),('v',1e-6),('a',1e-5)]:maximum_jump[k]=max(maximum_jump[k],e['jump'][k]);assert e['jump'][k]<tol
            if e['kind']=='commit':
                assert e['t']<e['start']-CONFIG['deadline_guard']-1e-10
                delivered=[f['stamp'] for f in frames if f['delivered_at'] is not None and f['delivered_at']<=e['t']+1e-9 and not f['dropped']]
                assert not delivered or max(delivered)<=e['map_stamp']+1e-9
        invalid=[];samples=0
        with Worker(row['backend'],a.out/row['folder'],route_policy='repair') as w:
            for i,q in enumerate(updates):
                cloud=dict(q['cloud']);cloud['binary_xyz_f32']=str((folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name).resolve())
                if 'sequence' in cloud:
                    f=frames[cloud['sequence']];assert not f['dropped'] and f['delivered_at'] is not None and cloud['stamp']<=f['delivered_at']<=q['t']+1e-8
                w.request(dict(op='update',t=q['t'],cloud=cloud));until=updates[i+1]['t'] if i+1<len(updates) else traj[-1,0]+.01
                subset=traj[(traj[:,0]>=q['t']-1e-9)&(traj[:,0]<until-1e-9)]
                if len(subset):
                    cells=w.request(dict(op='query',t=q['t'],points=subset[:,1:4].tolist()))['cells'];samples+=len(cells)
                    invalid.extend([dict(t=float(x[0]),cell=cell,position=x[1:4].tolist()) for x,cell in zip(subset,cells) if cell[0]!=0 or cell[1]])
        result=dict(folder=row['folder'],outcome=row['outcome'],samples=samples,invalid=invalid,final_speed=row['final_speed'],budget_rejections=row['budget_rejections']);results.append(result);save(a.out/'progress.json',results)
        print(json.dumps(dict(folder=row['folder'],invalid=len(invalid))),flush=True)
    save(a.out/'audit.json',dict(cases=len(results),model_outputs_recomputed=recomputed,maximum_handoff_jump=maximum_jump,executed_map_invalid_samples=sum(len(r['invalid']) for r in results),cases_detail=results,
        limitations=['10 ms executed samples, not continuous swept volume','Stop probes are finite and conditional on known map','Budget is cooperative; no real-time OS or hardware certification']))

if __name__=='__main__':main()
