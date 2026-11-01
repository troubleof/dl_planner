import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from navigation_common import save
from recovery_frontend import net_class,utilities,FORMAT


def actual_utility(y):return y[...,0]-3*y[...,1]-y[...,2]-.02*y[...,3]+.05*y[...,4]+.2*y[...,5]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    parts=[np.load(p/'dataset.npz') for p in a.datasets]
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    records=[r for p in a.datasets for r in json.loads((p/'records.json').read_text())]
    dataset_hashes={str(p):hashlib.sha256((p/'dataset.npz').read_bytes()).hexdigest() for p in a.datasets}
    ti=np.array([i for i,r in enumerate(records) if r['split']=='train']);vi=np.array([i for i,r in enumerate(records) if r['split']=='validation'])
    assert len(ti) and len(vi) and len(ti)+len(vi)==len(records)
    assert not ({records[i]['seed'] for i in ti}&{records[i]['seed'] for i in vi})
    x,s,c=[torch.from_numpy(data[k]) for k in ['image','state','candidates']]
    history=[]
    for target,seeds in [('short',[11]),('long',[11,22,33])]:
        y=torch.from_numpy(data[target]);q=actual_utility(y)
        unavailable=c[:,:,14::11].sum(dim=2)==0;unavailable[:,0]=False
        q=q.masked_fill(unavailable,-1e6)
        positive=y[ti][...,[1,2,4]].sum(dim=(0,1));negative=len(ti)*4-positive
        pos_weight=((negative+1)/(positive+1)).clamp(1,8)
        for seed in seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=net_class()()
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
            best=float('inf');epochs=[];begin=time.perf_counter()
            def loss_for(ix):
                pred=model(x[ix],s[ix],c[ix])
                regression=torch.nn.functional.smooth_l1_loss(pred[...,[0,3,5]],y[ix][...,[0,3,5]])
                classification=torch.nn.functional.binary_cross_entropy_with_logits(pred[...,[1,2,4]],y[ix][...,[1,2,4]],pos_weight=pos_weight)
                ranking=(-torch.softmax(q[ix]/.2,dim=1)*torch.log_softmax(utilities(pred)/.2,dim=1)).sum(dim=1).mean()
                return regression+.5*classification+.1*ranking,pred
            for epoch in range(1,121):
                model.train();order=rng.permutation(ti)
                for off in range(0,len(order),16):
                    loss,_=loss_for(order[off:off+16]);opt.zero_grad();loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
                model.eval()
                with torch.inference_mode():
                    val_loss,pred=loss_for(vi);value=utilities(pred)
                    # Empty proposal families are equivalent to repair; do not choose them.
                    empty=c[vi,:,14::11].sum(dim=2)==0
                    empty[:,0]=False;value[empty]=-1e6
                    chosen=value.argmax(dim=1)
                    chosen=torch.where(value.gather(1,chosen[:,None])[:,0]>=value[:,0]+.05,chosen,torch.zeros_like(chosen))
                    regret=(q[vi].max(dim=1).values-q[vi,chosen]).mean().item()
                    metric=regret+.05*val_loss.item()
                entry=dict(epoch=epoch,validation_regret=regret,validation_loss=val_loss.item(),selection_metric=metric,
                    validation_selected_failures=int(y[vi,chosen,1].sum()),validation_selected_stalls=int(y[vi,chosen,2].sum()))
                epochs.append(entry)
                if metric<best:
                    best=metric;selected=entry
                    torch.save(dict(format=FORMAT,target=target,seed=seed,epoch=epoch,state_dict=model.state_dict(),
                        dataset_sha256=dataset_hashes),a.out/f'{target}_s{seed}.pt')
            history.append(dict(target=target,seed=seed,parameters=sum(p.numel() for p in model.parameters()),training_states=len(ti),validation_states=len(vi),
                positive_counts=positive.tolist(),selected=selected,seconds=time.perf_counter()-begin,epochs=epochs))
            save(a.out/'training.json',history);print(json.dumps({k:v for k,v in history[-1].items() if k!='epochs'}),flush=True)
    save(a.out/'data_manifest.json',dict(dataset_sha256=dataset_hashes,protocols=[json.loads((p/'protocol.json').read_text()) for p in a.datasets]))


if __name__=='__main__':main()
