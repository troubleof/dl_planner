import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from direct_frontend import net_class,FORMAT
from navigation_common import save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1);d=np.load(a.dataset/'dataset.npz');rows=json.loads((a.dataset/'records.json').read_text())
    x,s=[torch.from_numpy(d[k].astype(np.float32)) for k in ['image','state']];ti=np.array([i for i,r in enumerate(rows) if r['split']=='train']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    histories=[]
    for seed in [11,22,33]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        path=a.models/f'geometry_s{seed}.pt';ckpt=torch.load(path,map_location='cpu',weights_only=True);model=net_class()();model.load_state_dict(ckpt['state_dict'])
        for name,param in model.named_parameters():param.requires_grad_(name.startswith('scorer.'))
        label=np.load(a.labels/f'labels_s{seed}.npz');target=torch.from_numpy(label['quality']);valid=torch.from_numpy(label['feasible'])
        with torch.inference_mode():
            predicted,_=model(x,s);assert np.max(np.abs(predicted.numpy()-label['parameters']))<2e-6
        optimizer=torch.optim.AdamW(model.scorer.parameters(),lr=1e-3,weight_decay=1e-4);best=float('inf');history=[];begin=time.perf_counter()
        def loss(ix):
            _,output=model(x[ix],s[ix]);score=output[...,0];logit=output[...,1]
            regression=torch.nn.functional.smooth_l1_loss(score,target[ix]);bce=torch.nn.functional.binary_cross_entropy_with_logits(logit,valid[ix])
            rank=(-torch.softmax(target[ix]/.25,dim=1)*torch.log_softmax(score/.25,dim=1)).sum(dim=1).mean()
            return regression+.4*bce+.1*rank,score,logit
        for epoch in range(1,161):
            model.train();order=rng.permutation(ti)
            for first in range(0,len(order),48):
                total,*_=loss(order[first:first+48]);optimizer.zero_grad();total.backward();optimizer.step()
            model.eval()
            with torch.inference_mode():
                total,score,logit=loss(vi);chosen=(score-2*(1-torch.sigmoid(logit))).argmax(dim=1)
                regret=(target[vi].max(dim=1).values-target[vi,chosen]).mean().item();passing=valid[vi,chosen].mean().item()
            metric=regret+.1*float(total);entry=dict(epoch=epoch,loss=float(total),regret=regret,first_choice_pass_rate=passing)
            history.append(entry)
            if metric<best:
                best=metric;selected=entry
                torch.save(dict(format=FORMAT,state_dict=model.state_dict(),seed=seed,epoch=epoch,stage='actual_output_scoring',geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    labels_sha256=hashlib.sha256((a.labels/f'labels_s{seed}.npz').read_bytes()).hexdigest()),a.models/f'direct_s{seed}.pt')
        histories.append(dict(seed=seed,selected=selected,seconds=time.perf_counter()-begin,history=history));save(a.models/'score_training.json',histories)
        print(json.dumps({k:v for k,v in histories[-1].items() if k!='history'}),flush=True)

if __name__=='__main__':main()
