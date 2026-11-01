"""Restore a causal logged prefix, then render all future scans from the branch."""
import json
from pathlib import Path
import numpy as np
from navigation_common import Curve,CONFIG

class PrefixMismatch(RuntimeError):
    """A restoration failure, before the experimental branch starts."""
    pass

def equivalent(a,b):
    for key in ['ok','reason','route_source','route_fallback','frontend']:
        if a.get(key)!=b.get(key):return False
    for key in ['route']:
        if (key in a)!=(key in b):return False
        if key in a and (np.shape(a[key])!=np.shape(b[key]) or not np.allclose(a[key],b[key],rtol=0,atol=1e-7)):return False
    for key in ['plan','backup_plan']:
        if (key in a)!=(key in b):return False
        if key in a:
            for field in ['start','duration','dt','controls']:
                if np.shape(a[key][field])!=np.shape(b[key][field]) or not np.allclose(a[key][field],b[key][field],rtol=0,atol=1e-7):return False
    return True

def restore(w,frontend,scene,channel,folder,index):
    folder=Path(folder);log=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()]
    target=log[index]['request'];assert target['op']=='plan'
    cut=index
    if index and log[index-1]['request']['op']=='context' and log[index-1]['request']['t']==target['t']:cut-=1
    last=None
    for item in log[:cut]:
        q=item['request']
        if 'cloud' in q:
            cloud=dict(q['cloud']);path=folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name
            cloud.pop('binary_xyz_f32');cloud.pop('count',None);cloud['points']=np.fromfile(path,dtype='<f4').reshape(-1,3).tolist()
            last=w.request(dict(op='update',t=q['t'],cloud=cloud))
        if q['op']=='plan':
            replay={k:v for k,v in q.items() if k not in ['cloud','neural_input','map_context','context_rpc_ms']}
            replay['_skip_frontend']=True;result=w.request(replay)
            if not equivalent(item['response'],result):
                raise PrefixMismatch(f"history differs at t={q['t']}: {item['response'].get('reason')} -> {result.get('reason')}")
    assert last is not None
    t=target['t'];trajectory=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1)
    frames=json.loads((folder/'sensor_frames.json').read_text())
    for frame in frames:
        if frame['stamp']>t+1e-9:break
        channel.capture(scene,frame['origin'],frame['stamp'])
        sample=trajectory[np.argmin(np.abs(trajectory[:,0]-frame['stamp']))]
        assert abs(sample[0]-frame['stamp'])<1e-8
        state=dict(p=sample[1:4].tolist(),v=sample[4:7].tolist(),a=sample[7:10].tolist())
        for _,sequence,cloud in channel.queue:
            if sequence==channel.next_seq-1:cloud['capture_state']=state
    latest=channel.deliver(t)
    for f,original in zip(channel.records,frames):
        if original['delivered_at'] is not None and original['delivered_at']<=t+1e-9:f['delivered_at']=original['delivered_at']
        f['mapped_at']=original['mapped_at'] if f['stamp']<=last['map_stamp']+1e-9 else None
    schedule=[Curve(p) for p in target['execution_schedule']]
    return t,latest,last['map_stamp'],last['map_version'],schedule,dict(folder=str(folder),request_index=index,cut_index=cut,t=t,map_stamp=last['map_stamp'])
