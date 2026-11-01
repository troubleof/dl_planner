"""Version 2 residual targets from preserved version 1 fitted teacher controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from direct_frontend import BASE,PARAMS,K
from navigation_common import save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    d=np.load(a.source/'dataset.npz');old=d['target'];state=d['state'];mask=d['mask'];target=np.zeros((len(old),K,PARAMS),np.float32);target[...,0]=old[...,0]
    for i in range(len(old)):
        for j in range(K):
            if not mask[i,j]:continue
            T=np.exp(float(old[i,j,0]));points=old[i,j,1:].reshape(7,3)*3
            base=BASE@np.stack([points[-1],state[i,23:26]*3*T,state[i,26:29]*6*T*T])
            target[i,j,1:13]=((points[:6,:2]-base[:,:2])/3).ravel();target[i,j,13:15]=points[-1,:2]/3
    rows=json.loads((a.source/'records.json').read_text());replays=json.loads((a.source/'replays.json').read_text())
    for row in rows:row['cloud']['binary_xyz_f32']='../'+a.source.name+'/'+row['cloud']['binary_xyz_f32']
    for replay in replays.values():
        for event in replay['sequence']:
            if 'cloud' in event:event['cloud']['binary_xyz_f32']='../'+a.source.name+'/'+event['cloud']['binary_xyz_f32']
    np.savez_compressed(a.out/'dataset.npz',image=d['image'],state=state,target=target,mask=mask)
    save(a.out/'records.json',rows);save(a.out/'replays.json',replays)
    protocol=json.loads((a.source/'protocol.json').read_text());protocol.update(parent_dataset=str(a.source),parent_sha256=hashlib.sha256((a.source/'dataset.npz').read_bytes()).hexdigest(),
        parameterization='duration, six planar residuals about a PVA-aware minimum-jerk base, endpoint XY; analytic vertical component',
        dataset_sha256=hashlib.sha256((a.out/'dataset.npz').read_bytes()).hexdigest())
    save(a.out/'protocol.json',protocol);print(json.dumps(dict(states=len(rows),converted=True)),flush=True)

if __name__=='__main__':main()
