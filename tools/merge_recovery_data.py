import argparse,hashlib,json
from pathlib import Path
import numpy as np
from navigation_common import save
from recovery_frontend import HORIZONS

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[];arrays={};events=[];sources=[]
    for folder in a.inputs:
        data=np.load(folder/'dataset.npz');source_rows=json.loads((folder/'records.json').read_text());offset=len(rows)
        for row in source_rows:rows.append(dict(row,index=row['index']+offset,dataset_source=str(folder.resolve())))
        for key in data.files:arrays.setdefault(key,[]).append(data[key])
        events.extend(dict(e,state=e['state']+offset) for e in json.loads((folder/'future_invalidations.json').read_text()))
        sources.append(dict(path=str(folder.resolve()),sha256=hashlib.sha256((folder/'dataset.npz').read_bytes()).hexdigest()))
    arrays={k:np.concatenate(v) for k,v in arrays.items()};corrected=0
    for ev in events:
        for h,delta in enumerate(HORIZONS,1):
            i,j=ev['state'],ev['candidate']
            if ev['observed_at']<=rows[i]['query']['t']+delta+1e-9:
                corrected+=int(arrays['mask'][i,j,h]==0);arrays['labels'][i,j,h]=0;arrays['mask'][i,j,h]=1
    train={r['seed'] for r in rows if r['split']=='train'};val={r['seed'] for r in rows if r['split']=='validation'};assert not train&val
    np.savez_compressed(a.out/'dataset.npz',**arrays);save(a.out/'records.json',rows);save(a.out/'future_invalidations.json',events)
    save(a.out/'protocol.json',dict(states=len(rows),sources=sources,train_maps=sorted(train),validation_maps=sorted(val),fresh_test_maps=[5401,5402,5403,5404],old_test_reassigned_states=sum(r.get('original_split')=='test' for r in rows),censor_corrections=corrected,current_valid=int(arrays['labels'][:,:,0].sum()),future_negative=int(((arrays['labels'][:,:,0:1]>.5)&(arrays['labels'][:,:,1:]<.5)&(arrays['mask'][:,:,1:]>0)).sum()),dataset_sha256=hashlib.sha256((a.out/'dataset.npz').read_bytes()).hexdigest(),note='Historical diagnosed test failures are now training/validation, and MUST NOT be called independent tests of these new models. Previous source/logs unchanged. Fresh 5401-5404 reserved.'))
    print(json.dumps(json.loads((a.out/'protocol.json').read_text())),flush=True)

if __name__=='__main__':main()
