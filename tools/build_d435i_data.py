"""Render only front-camera observations on old training poses; revalidate teachers."""
import argparse,copy,hashlib,json,re,time
from pathlib import Path
import numpy as np
from navigation_common import Worker,make_scene,save
from d435i_adapter import D435iSensor,camera_observation,CAMERA,camera_scene
from direct_frontend import decode

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);(a.out/'clouds').mkdir()
    old=json.loads((a.dataset/'records.json').read_text());replays=json.loads((a.dataset/'replays.json').read_text());data=np.load(a.dataset/'dataset.npz')
    sensor=D435iSensor();rows=[];images=[];states=[];targets=[];masks=[];newreplays={};counts={};begin=time.perf_counter();frame=0
    for case,rep in replays.items():
        seed=int(re.search(r'forest_(\d+)',case)[1]);scene=camera_scene('forest',seed);sequence=[];latest=None;reference=None
        with Worker(rep['backend'],a.out/'workers'/case) as w:
            for event in rep['sequence']:
                if 'sample' not in event:
                    c=event['cloud'];latest=sensor.capture(scene,c['origin'],c['stamp'])
                    if 'capture_state' in c:latest['capture_state']=c['capture_state']
                    frame+=1;prefix=a.out/'clouds'/f'{frame:05d}';np.asarray(latest['points'],dtype='<f4').reshape(-1,3).tofile(prefix.with_suffix('.bin'))
                    reference={k:v for k,v in latest.items() if k not in ['points','depth_grid_world','depth_valid']};reference.update(binary_xyz_f32=str(prefix.with_suffix('.bin').resolve()),count=len(latest['points']),depth_metadata_json=str(prefix.with_suffix('.json').resolve()))
                    save(prefix.with_suffix('.json'),{k:latest[k] for k in ['depth_grid_world','depth_valid']})
                    update=dict(op='update',t=event['t'],cloud=reference);w.request(update);sequence.append(update);continue
                i=event['sample'];original=old[i];q=original['query'];teacher,_=decode(q,data['target'][i],np.zeros((15,2)))
                teacher=[c for c in teacher if data['mask'][i,c['id']]>0]
                checked=w.request(dict(q,op='evaluate_direct',direct_plans=teacher,forbid_optimizer=True));mask=np.zeros(15,np.float32)
                for b in checked['branches']:
                    counts[b['reason']]=counts.get(b['reason'],0)+1
                    if b['ok']:mask[b['id']]=1
                variants=[('original',None)]
                if i%4==0 and not original['near_goal']:variants += [('left',np.pi/2),('reverse',np.pi),('right',-np.pi/2)]
                for name,angle in variants:
                    row=copy.deepcopy(original);row['index']=len(rows);row['parent_index']=i;row['motion_variant']=name;row['cloud']=reference
                    if angle is not None:
                        c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                        for key in ['v','a']:row['query']['start_state'][key]=(R@np.asarray(row['query']['start_state'][key])).tolist()
                    im,st=camera_observation(latest,row['query'],row['backend']);images.append(im);states.append(st);targets.append(data['target'][i]);masks.append(mask if angle is None else np.zeros(15,np.float32))
                    sequence.append(dict(sample=len(rows)));rows.append(row)
        newreplays[case]=dict(backend=rep['backend'],sequence=sequence)
        print(json.dumps(dict(case=case,states=len(rows),valid_teachers=int(np.sum(masks)),seconds=time.perf_counter()-begin)),flush=True)
    np.savez_compressed(a.out/'dataset.npz',image=images,state=states,target=targets,mask=masks);save(a.out/'records.json',rows);save(a.out/'replays.json',newreplays)
    save(a.out/'protocol.json',dict(states=len(rows),original_states=len(old),valid_teacher_curves=int(np.sum(masks)),teacher_checks=counts,camera=CAMERA,
        source_dataset_sha256=hashlib.sha256((a.dataset/'dataset.npz').read_bytes()).hexdigest(),
        expert='Frozen EGO teacher curves, admitted only after revalidation on front-camera fused map and stop check; no new hidden observation enters input',
        limitation='Training poses inherited from old panorama-policy rollouts; not camera-policy on-policy trajectories. Counterfactual v/a are offline diagnostics.'))

if __name__=='__main__':main()
