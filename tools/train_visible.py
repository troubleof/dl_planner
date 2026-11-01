"""Timing/geometry imitation, then ranking on reevaluated frozen model curves."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from visible_frontend import FORMAT,network_class
from navigation_common import save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--stage',choices=['geometry','scores'],required=True);ap.add_argument('--geometry',type=Path);ap.add_argument('--labels',type=Path);ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);ap.add_argument('--epochs',type=int,default=90);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    data=np.load(a.dataset/'dataset.npz');rows=json.loads((a.dataset/'records.json').read_text());tensors={k:torch.from_numpy(data[k].astype(np.float32)) for k in data.files};ti=np.array([i for i,r in enumerate(rows) if r['split']=='train']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);assert not ({rows[i]['seed'] for i in ti}&{rows[i]['seed'] for i in vi});histories=[]
    dataset_hash=hashlib.sha256((a.dataset/'dataset.npz').read_bytes()).hexdigest()
    def forward(model,ix):return model(*(tensors[k][ix] for k in ['image','state','seed','features']))
    for seed in a.seeds:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=network_class()();begin=time.perf_counter();history=[];best=float('inf');extra={}
        if a.stage=='scores':
            path=a.geometry/f'geometry_s{seed}.pt';ck=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);label=np.load(a.labels/f's{seed}.npz');target=torch.from_numpy(label['quality']-label['feasible']*(.35*label['risk']+.35*(1-label['recovery'])));valid=torch.from_numpy(label['feasible'])
            for name,p in model.named_parameters():p.requires_grad_(name.startswith('scorer.'))
            with torch.inference_mode():
                for first in range(0,len(rows),64):
                    raw,_=forward(model,np.arange(first,min(first+64,len(rows))));assert np.max(np.abs(raw.numpy()-label['parameters'][first:first+64]))<3e-6
            extra=dict(geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),labels_sha256=hashlib.sha256((a.labels/f's{seed}.npz').read_bytes()).hexdigest())
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=8e-4,weight_decay=1e-4)
        def loss(ix):
            raw,pred=forward(model,ix)
            if a.stage=='geometry':
                mask=tensors['mask'][ix];den=mask.sum().clamp_min(1);diff=raw-tensors['target'][ix];return ((diff[...,0].square()+30*diff[...,1:].square().mean(dim=-1))*mask).sum()/den,{}
            score=pred[...,0];logit=pred[...,1];reg=torch.nn.functional.smooth_l1_loss(score,target[ix]);bce=torch.nn.functional.binary_cross_entropy_with_logits(logit,valid[ix]);rank=(-torch.softmax(target[ix]/.25,dim=1)*torch.log_softmax(score/.25,dim=1)).sum(dim=1).mean();chosen=(score-2*(1-torch.sigmoid(logit))).argmax(dim=1)
            return reg+.4*bce+.1*rank,dict(regret=float((target[ix].max(dim=1).values-target[ix,chosen]).mean().detach()),first_choice_pass_rate=float(valid[ix,chosen].mean()))
        for epoch in range(1,a.epochs+1):
            model.train();order=rng.permutation(ti)
            for first in range(0,len(order),32):
                value,_=loss(order[first:first+32]);opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            model.eval()
            with torch.inference_mode():val,metrics=loss(vi)
            entry=dict(epoch=epoch,loss=float(val),**metrics);history.append(entry);criterion=metrics.get('regret',0)+float(val)*(.1 if metrics else 1.)
            if criterion<best:
                best=criterion;selected=entry;name='geometry' if a.stage=='geometry' else 'visible';torch.save(dict(format=FORMAT,seed=seed,state_dict=model.state_dict(),epoch=epoch,sensor='d435i',input_shape=[6,60,106],dataset_sha256=dataset_hash,stage=a.stage,**extra),a.out/f'{name}_s{seed}.pt')
        histories.append(dict(seed=seed,stage=a.stage,selected=selected,train=len(ti),validation=len(vi),seconds=time.perf_counter()-begin,history=history));save(a.out/'training.json',histories);print(json.dumps({k:v for k,v in histories[-1].items() if k!='history'}),flush=True)

if __name__=='__main__':main()
