"""Matched ranking ablations; frozen geometry, expanded motion-state dataset."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from motion_frontend import net_class,FORMAT,SPEC
from navigation_common import save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False);d=np.load(a.dataset/'dataset.npz');rows=json.loads((a.dataset/'records.json').read_text());manifest=json.loads((a.labels/'protocol.json').read_text())['models']
    ti=np.array([i for i,r in enumerate(rows) if r['split']=='train']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);histories=[]
    assert not ({rows[i]['seed'] for i in ti}&{rows[i]['seed'] for i in vi})
    for variant,seed in [('control',11),('features',11),('labels',11),('both',11),('both',22),('both',33)]:
        bank='features' if variant in ['features','both'] else 'base';key=f'{bank}{seed}';spec=manifest[key];channels=spec['channels']
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=net_class(channels)();ck=torch.load(spec['path'],map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict'])
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('scorer.'))
        x=torch.from_numpy(d['image'][:,:channels]);s=torch.from_numpy(d['state']);lab=np.load(a.labels/(key+'.npz'));target=lab['quality'].copy()
        if variant in ['labels','both']:target-=lab['feasible']*(SPEC['risk_weight']*lab['risk']+SPEC['recovery_weight']*(1-lab['recovery']))
        target=torch.from_numpy(target);valid=torch.from_numpy(lab['feasible'])
        with torch.inference_mode():raw,_=model(x,s);assert np.max(np.abs(raw.numpy()-lab['parameters']))<3e-6
        opt=torch.optim.AdamW(model.scorer.parameters(),lr=1e-3,weight_decay=1e-4);hist=[];best=float('inf');begin=time.perf_counter()
        def loss(ix):
            _,pred=model(x[ix],s[ix]);score=pred[...,0];logit=pred[...,1]
            reg=torch.nn.functional.smooth_l1_loss(score,target[ix]);bce=torch.nn.functional.binary_cross_entropy_with_logits(logit,valid[ix])
            rank=(-torch.softmax(target[ix]/.25,dim=1)*torch.log_softmax(score/.25,dim=1)).sum(dim=1).mean()
            return reg+.4*bce+.1*rank,score,logit
        for epoch in range(1,161):
            model.train();order=rng.permutation(ti)
            for first in range(0,len(order),48):
                total,*_=loss(order[first:first+48]);opt.zero_grad();total.backward();opt.step()
            model.eval()
            with torch.inference_mode():
                total,score,logit=loss(vi);chosen=(score-2*(1-torch.sigmoid(logit))).argmax(dim=1)
                regret=(target[vi].max(dim=1).values-target[vi,chosen]).mean().item();passing=valid[vi,chosen].mean().item()
            entry=dict(epoch=epoch,loss=float(total),regret=regret,first_choice_pass_rate=passing);hist.append(entry);metric=regret+.1*float(total)
            if metric<best:
                best=metric;selected=entry;torch.save(dict(format=FORMAT,channels=channels,variant=variant,seed=seed,epoch=epoch,state_dict=model.state_dict(),sensor='d435i',input_shape=[channels,60,106],
                    geometry_sha256=spec['sha256'],labels_sha256=hashlib.sha256((a.labels/(key+'.npz')).read_bytes()).hexdigest(),risk_spec=SPEC),a.out/f'{variant}_s{seed}.pt')
        histories.append(dict(variant=variant,seed=seed,selected=selected,train=len(ti),validation=len(vi),seconds=time.perf_counter()-begin,history=hist));save(a.out/'training.json',histories)
        print(json.dumps({k:v for k,v in histories[-1].items() if k!='history'}),flush=True)

if __name__=='__main__':main()
