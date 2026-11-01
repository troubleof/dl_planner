import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from navigation_common import save
from context_frontend import net_class,predicted_utility

def target_utility(y):return y[...,0]-2*y[...,1]-.05*y[...,2]-.08*y[...,3]+.05*y[...,4]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--targets',nargs='+',default=['multistep']);ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);ds=[np.load(p/'dataset.npz') for p in a.datasets];records=sum([json.loads((p/'records.json').read_text()) for p in a.datasets],[])
    x={k:torch.from_numpy(np.concatenate([d[k] for d in ds])) for k in ['rays','local_map','state']}
    ti=np.array([i for i,r in enumerate(records) if r['split']=='train']);vi=np.array([i for i,r in enumerate(records) if r['split']=='validation']);assert len(ti) and len(vi)
    history=[]
    for target in a.targets:
        y=torch.from_numpy(np.concatenate([d[target] for d in ds]));true_value=target_utility(y)
        for seed in a.seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=net_class()();optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3);best=float('inf');epochs=[];begin=time.perf_counter()
            for epoch in range(1,121):
                model.train();order=rng.permutation(ti)
                for offset in range(0,len(order),32):
                    ids=order[offset:offset+32];p=model(*(x[k][ids] for k in ['rays','local_map','state']));truth=y[ids]
                    loss=torch.nn.functional.smooth_l1_loss(p[...,0],truth[...,0])+.05*torch.nn.functional.smooth_l1_loss(p[...,2],truth[...,2])+.1*torch.nn.functional.smooth_l1_loss(p[...,3],truth[...,3])
                    loss+=.5*torch.nn.functional.binary_cross_entropy_with_logits(p[...,1],truth[...,1],pos_weight=torch.tensor(3.))+.1*torch.nn.functional.binary_cross_entropy_with_logits(p[...,4],truth[...,4])
                    desired=torch.softmax(true_value[ids]/.15,dim=1);loss+=.15*(-desired*torch.log_softmax(predicted_utility(p)/.15,dim=1)).sum(dim=1).mean()
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                model.eval()
                with torch.inference_mode():
                    p=model(*(x[k][vi] for k in ['rays','local_map','state']));u=predicted_utility(p);chosen=u.argmax(dim=1);chosen=torch.where(u[torch.arange(len(vi)),chosen]>=u[:,4]+.05,chosen,4)
                    regret=(true_value[vi].max(dim=1).values-true_value[vi,chosen]).mean().item();risk=y[vi,chosen,1].mean().item();fallback=(chosen==4).float().mean().item()
                    error=torch.nn.functional.smooth_l1_loss(u,true_value[vi]).item();metric=regret+2*risk+.1*error
                entry=dict(epoch=epoch,validation_regret=regret,validation_selected_failure=risk,validation_repair_fraction=fallback,validation_value_loss=error);epochs.append(entry)
                if metric<best:
                    best=metric;selected=dict(entry);torch.save(dict(state_dict=model.state_dict(),seed=seed,target=target,epoch=epoch),a.out/f'{target}_s{seed}.pt')
            result=dict(seed=seed,target=target,parameters=sum(p.numel() for p in model.parameters()),train_states=len(ti),validation_states=len(vi),selected=selected,history=epochs,seconds=time.perf_counter()-begin)
            history.append(result);save(a.out/'training.json',history);print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    save(a.out/'data_manifest.json',[dict(path=str(p),sha256=hashlib.sha256((p/'dataset.npz').read_bytes()).hexdigest()) for p in a.datasets])
if __name__=='__main__':main()
