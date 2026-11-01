import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from direct_frontend import net_class,controls_torch,FORMAT
from navigation_common import save,CONFIG

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1);a.out.mkdir(parents=True,exist_ok=False)
    d=np.load(a.dataset/'dataset.npz');rows=json.loads((a.dataset/'records.json').read_text())
    x,s,target,mask=[torch.from_numpy(d[k].astype(np.float32)) for k in ['image','state','target','mask']]
    ti=np.array([i for i,r in enumerate(rows) if r['split']=='train']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    assert not ({rows[i]['seed'] for i in ti}&{rows[i]['seed'] for i in vi})
    histories=[]
    def loss(model,index):
        pred,_=model(x[index],s[index]);m=mask[index];den=m.sum().clamp_min(1)
        shape=((pred[...,1:]-target[index,...,1:]).square().mean(dim=-1)*m).sum()/den
        timing=((pred[...,0]-target[index,...,0]).square()*m).sum()/den
        q,h=controls_torch(pred,s[index]);v=q.diff(dim=2)/h[:,:,None,None];acc=v.diff(dim=2)/h[:,:,None,None]
        vb=v.norm(dim=-1).amax(dim=2);ab=acc.norm(dim=-1).amax(dim=2)
        physical=((torch.relu(vb-2.8)/3).square()+(torch.relu(ab-5.5)/6).square())
        physical=(physical*m).sum()/den
        return 20*shape+timing+.5*physical,dict(shape=float(shape),time=float(timing),physics=float(physical),
            masked_dynamic_pass=float((((vb<=CONFIG['max_velocity'])&(ab<=CONFIG['max_acceleration'])).float()*m).sum()/den))
    for seed in [11,22,33]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=net_class()()
        with torch.no_grad():
            mean=(target[ti]*mask[ti,:,None]).sum(dim=0)/mask[ti].sum(dim=0)[:,None].clamp_min(1)
            model.geometry.weight.zero_();model.geometry.bias.copy_(mean.flatten())
        optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4);best=float('inf');history=[];start=time.perf_counter()
        for epoch in range(1,181):
            model.train();order=rng.permutation(ti)
            for first in range(0,len(order),32):
                total,_=loss(model,order[first:first+32]);optimizer.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            model.eval()
            with torch.inference_mode():total,metrics=loss(model,vi)
            history.append(dict(epoch=epoch,loss=float(total),**metrics))
            if float(total)<best:
                best=float(total);selected=history[-1]
                torch.save(dict(format=FORMAT,state_dict=model.state_dict(),seed=seed,epoch=epoch,stage='geometry',
                    dataset_sha256=hashlib.sha256((a.dataset/'dataset.npz').read_bytes()).hexdigest()),a.out/f'geometry_s{seed}.pt')
        histories.append(dict(seed=seed,selected=selected,training_samples=len(ti),validation_samples=len(vi),parameters=sum(p.numel() for p in model.parameters()),
            wall_seconds=time.perf_counter()-start,history=history))
        save(a.out/'geometry_training.json',histories);print(json.dumps({k:v for k,v in histories[-1].items() if k!='history'}),flush=True)

if __name__=='__main__':main()
