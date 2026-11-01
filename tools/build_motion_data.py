"""Reuse frozen teachers, add observation-only same-state motion counterfactuals."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from motion_frontend import observation,SPEC
from navigation_common import save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);source=np.load(a.dataset/'dataset.npz');old=json.loads((a.dataset/'records.json').read_text())
    replay=json.loads((a.dataset/'replays.json').read_text());rows=[];images=[];states=[];targets=[];masks=[];replays={}
    for case,rep in replay.items():
        sequence=[]
        for event in rep['sequence']:
            if 'sample' not in event:
                ev=copy.deepcopy(event);ev['cloud']['binary_xyz_f32']=str((a.dataset/ev['cloud']['binary_xyz_f32']).resolve());sequence.append(ev);continue
            i=event['sample'];row=old[i];cloud=dict(row['cloud']);cloud['binary_xyz_f32']=str((a.dataset/cloud['binary_xyz_f32']).resolve())
            cloud['points']=np.fromfile(cloud['binary_xyz_f32'],dtype='<f4').reshape(-1,3)
            variants=[('original',None)]
            if i%4==0 and not row['near_goal']:
                variants += [('left',np.pi/2),('reverse',np.pi),('right',-np.pi/2)]
            for name,angle in variants:
                r=copy.deepcopy(row);r['index']=len(rows);r['parent_index']=i;r['motion_variant']=name;r['cloud']={k:v for k,v in cloud.items() if k!='points'}
                if angle is not None:
                    c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
                    for k in ['v','a']:r['query']['start_state'][k]=(rot@np.asarray(r['query']['start_state'][k])).tolist()
                im,st=observation(cloud,r['query'],r['backend']);images.append(im);states.append(st)
                targets.append(source['target'][i]);masks.append(source['mask'][i] if angle is None else np.zeros(15,np.float32))
                sequence.append(dict(sample=len(rows)));rows.append(r)
        replays[case]=dict(backend=rep['backend'],sequence=sequence)
    np.savez_compressed(a.out/'dataset.npz',image=images,state=states,target=targets,mask=masks)
    save(a.out/'records.json',rows);save(a.out/'replays.json',replays)
    save(a.out/'protocol.json',dict(source=str(a.dataset.resolve()),source_sha256=hashlib.sha256((a.dataset/'dataset.npz').read_bytes()).hexdigest(),
        states=len(rows),original_states=len(old),counterfactual_states=len(rows)-len(old),spec=SPEC,
        training_seeds=sorted({r['seed'] for r in rows if r['split']=='train'}),validation_seeds=sorted({r['seed'] for r in rows if r['split']=='validation'}),
        counterfactual='only current/handoff v,a rotated; no changed/future cloud, no teacher geometry label, ranking labels revalidated; not physically executed prefixes'))
    print(json.dumps(dict(states=len(rows),original=len(old))),flush=True)

if __name__=='__main__':main()
