"""Same original 471 teacher states, architecture/loss/epochs as frozen baseline."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from motion_frontend import net_class,FORMAT
from direct_frontend import controls_torch
from navigation_common import save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);ap.add_argument('--channels',type=int,choices=[3,6],default=6);a=ap.parse_args()
    torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False)
    d=np.load(a.dataset/'dataset.npz');rows=json.loads((a.dataset/'records.json').read_text());x,s,target,mask=[torch.from_numpy(d[k].astype(np.float32)) for k in ['image','state','target','mask']]
    x=x[:,:a.channels]
    ti=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['motion_variant']=='original']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation' and r['motion_variant']=='original'])
    histories=[]
    def loss(model,ix):
        raw,_=model(x[ix],s[ix]);m=mask[ix];den=m.sum().clamp_min(1)
        shape=((raw[...,1:]-target[ix,...,1:]).square().mean(dim=-1)*m).sum()/den;timing=((raw[...,0]-target[ix,...,0]).square()*m).sum()/den
        q,h=controls_torch(raw,s[ix]);v=q.diff(dim=2)/h[:,:,None,None];acc=v.diff(dim=2)/h[:,:,None,None]
        vb=v.norm(dim=-1).amax(dim=2);ab=acc.norm(dim=-1).amax(dim=2)
        physical=(((torch.relu(vb-2.8)/3).square()+(torch.relu(ab-5.5)/6).square())*m).sum()/den
        return 20*shape+timing+.5*physical
    for seed in a.seeds:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=net_class(a.channels)()
        with torch.no_grad():
            mean=(target[ti]*mask[ti,:,None]).sum(dim=0)/mask[ti].sum(dim=0)[:,None].clamp_min(1);model.geometry.weight.zero_();model.geometry.bias.copy_(mean.flatten())
        opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4);best=float('inf');hist=[];begin=time.perf_counter()
        for epoch in range(1,181):
            model.train();order=rng.permutation(ti)
            for first in range(0,len(order),32):
                total=loss(model,order[first:first+32]);opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            model.eval()
            with torch.inference_mode():val=float(loss(model,vi))
            hist.append(dict(epoch=epoch,loss=val))
            if val<best:
                best=val;selected=epoch;torch.save(dict(format=FORMAT,channels=a.channels,seed=seed,variant='features',state_dict=model.state_dict(),epoch=epoch,
                    dataset_sha256=hashlib.sha256((a.dataset/'dataset.npz').read_bytes()).hexdigest()),a.out/f'geometry_s{seed}.pt')
        histories.append(dict(seed=seed,selected_epoch=selected,loss=best,seconds=time.perf_counter()-begin,train=len(ti),validation=len(vi),history=hist));save(a.out/'geometry_training.json',histories)
        print(json.dumps({k:v for k,v in histories[-1].items() if k!='history'}),flush=True)

if __name__=='__main__':main()
