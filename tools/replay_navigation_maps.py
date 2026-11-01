#!/usr/bin/env python3
"""Identical captured frames to both backends, separate from closed-loop trials."""
import argparse
from navigation_common import *

def replay(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    scene=make_scene('forest',77);sensor=Sensor();frames=[]
    # This is a prescribed sensor replay, not a successfully planned flight.
    for x in np.linspace(-6.,6.,13):
        for _ in range(3):frames.append(sensor.capture(scene,[float(x),-3.8,2.5],len(frames)*.2))
    xy=np.array(np.meshgrid(np.arange(-8.95,9,.1),np.arange(-4.95,5,.1))).reshape(2,-1).T
    probes=np.column_stack([xy,np.full(len(xy),2.55)]).tolist();np.save(out/'query_points.npy',probes)
    save(out/'scene.json',scene);save(out/'config.json',CONFIG)
    save(out/'frame_manifest.json',[dict(stamp=f['stamp'],origin=f['origin'],point_sha256=hashlib.sha256(np.asarray(f['points'],dtype='<f4').tobytes()).hexdigest()) for f in frames])
    results={};grids={}
    for backend in ['ego','rog']:
        times=[];snapshots=[]
        with Worker(backend,out/backend) as w:
            for i,f in enumerate(frames):
                r=w.request(dict(op='update',t=f['stamp'],cloud=f));times.append(r)
                if i in [2,11,20,29,38]:snapshots.append(w.request(dict(op='query',t=f['stamp'],points=probes))['cells'])
        grids[backend]=np.asarray(snapshots,dtype=np.int8);np.save(out/f'{backend}_cells.npy',grids[backend]);save(out/f'{backend}_timings.json',times)
        results[backend]={'frames':len(frames),'map_update_ms':{f'p{q}':float(np.percentile([r['map_update_ms'] for r in times],q)) for q in [50,95,99]},'rpc_ms':{f'p{q}':float(np.percentile([r['rpc_ms'] for r in times],q)) for q in [50,95,99]}}
    results['query_disagreement']=[dict(frame=i,raw_state_fraction=float(np.mean(a[:,0]!=b[:,0])),inflated_occupancy_fraction=float(np.mean(a[:,1]!=b[:,1])),ego_unknown=int(np.sum(a[:,0]==2)),rog_unknown=int(np.sum(b[:,0]==2))) for i,a,b in zip([2,11,20,29,38],grids['ego'],grids['rog'])]
    results['interpretation']='Same float32 points and capture poses; native prior and ray update algorithms differ. No claim of equivalent posterior maps or closed-loop speed advantage.'
    save(out/'summary.json',results);print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();replay(a.out)
