"""Equal-data/seed architecture ablation. Validation-only model selection."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from navigation_common import save
from recovery_frontend import FORMAT,HORIZONS,network_class,probabilities
from recovery_dataset import load_dataset

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--kinds',nargs='+',default=['mlp','spatial','temporal']);ap.add_argument('--seeds',nargs='+',type=int,default=[11,22,33]);ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--current-only',action='store_true');ap.add_argument('--tag',default='');ap.add_argument('--base-only',action='store_true',help='Use the verified 827-row base prefix; no second dataset copy.');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    data,rows,protocol,repaired=load_dataset(a.dataset,a.base_only);train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);assert len(train) and len(val)
    # Preserve raw dataset v2, but distinguish observed failure from true
    # censoring. Save exact effective targets and hashes used for all models.
    np.savez_compressed(a.out/'effective_supervision.npz',labels=data['labels'],mask=data['mask'])
    tensors={k:torch.from_numpy(data[k]) for k in ['image','previous','state','features','labels','mask']};risk=tensors['mask'].clone();risk[:,:,1:]*=tensors['labels'][:,:,:-1]*tensors['mask'][:,:,:-1]
    # A padded duplicate candidate must not multiply its training influence.
    weights=np.ones(data['seed'].shape[:2],np.float32)
    for i,row in enumerate(data['seed']):
        _,inverse,count=np.unique(np.round(row,6),axis=0,return_inverse=True,return_counts=True);weights[i]=1/count[inverse]
    risk*=torch.from_numpy(weights[:,:,None])
    if a.current_only:risk[:,:,1:]=0
    def loss(logits,idx):
        target=tensors['labels'][idx];mask=risk[idx];weight=torch.where(target>.5,1.,4.)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none');return (bce*mask*weight).sum()/(mask*weight).sum().clamp_min(1)
    def evaluate(model,indices):
        values=[];raw=[]
        with torch.inference_mode():
            for batch in np.array_split(indices,max(1,int(np.ceil(len(indices)/32)))):
                logits=model(*(tensors[k][batch] for k in ['image','previous','state','features']));values.append(float(loss(logits,batch))*len(batch));raw.append(logits.numpy())
        pred=probabilities(np.concatenate(raw))
        if a.current_only:pred=np.repeat(pred[:,:,:1],4,axis=-1)
        labels=data['labels'][indices];mask=data['mask'][indices];valid=labels[:,:,0]>.5;future=mask[:,:,-1]>0;negative=valid&future&(labels[:,:,-1]<.5)
        choice=np.argmax(data['features'][indices,:,-1]-.8*(1-pred[:,:,-1]),axis=1);rr=np.arange(len(indices));known=future[rr,choice]&valid[rr,choice]
        return dict(loss=sum(values)/len(indices),brier=float((((pred-labels)**2)*mask).sum()/mask.sum()),current_valid_future_negative=int(negative.sum()),negative_recall_at_half=float((negative&(pred[:,:,-1]<.5)).sum()/max(1,negative.sum())),known_selected=int(known.sum()),selected_future_invalid=int((known&(labels[rr,choice,-1]<.5)).sum())),pred
    meta=dict(format=FORMAT,dataset=str(a.dataset.resolve()),dataset_sha256=hashlib.sha256((a.dataset/'dataset.npz').read_bytes()).hexdigest(),train_states=len(train),validation_states=len(val),kinds=a.kinds,seeds=a.seeds,epochs=a.epochs,optimizer='Adam lr=0.001 batch32 weight_decay=1e-5 clip_grad=2',loss='Conditional survival BCE, invalid transition weight4, masked right censoring',selection='minimum validation weighted hazard loss; no test map selection',risk_weight=.8,geometry='fixed exact-PVA timed candidate seeds',horizons=HORIZONS)
    meta.update(current_only=a.current_only,base_only=a.base_only,tag=a.tag,duplicate_weights='inverse exact rounded candidate multiplicity',observed_failure_censor_corrections=repaired,effective_supervision_sha256=hashlib.sha256((a.out/'effective_supervision.npz').read_bytes()).hexdigest())
    save(a.out/'training_protocol.json',meta);summary=[]
    for kind in a.kinds:
        for seed in a.seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=network_class(kind)();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5);best=float('inf');history=[];best_state=None;start=time.perf_counter()
            for epoch in range(a.epochs):
                model.train();indices=rng.permutation(train);values=[]
                for batch in np.array_split(indices,max(1,int(np.ceil(len(indices)/32)))):
                    opt.zero_grad();logits=model(*(tensors[k][batch] for k in ['image','previous','state','features']));value=loss(logits,batch);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();values.append(float(value.detach()))
                model.eval();score,_=evaluate(model,val);history.append(dict(epoch=epoch,train_loss=float(np.mean(values)),validation=score))
                if score['loss']<best:best=score['loss'];best_state=copy.deepcopy(model.state_dict());chosen=epoch
            model.load_state_dict(best_state);score,pred=evaluate(model,val);name=f'{kind}{a.tag}{seed}';path=a.out/(name+'.pt');torch.save(dict(format=FORMAT,kind=kind,seed=seed,state_dict=best_state,epoch=chosen,current_only=a.current_only,dataset_sha256=meta['dataset_sha256']),path);save(a.out/(name+'_history.json'),history);np.savez_compressed(a.out/(name+'_validation.npz'),indices=val,prediction=pred)
            row=dict(name=name,seed=seed,kind=kind,chosen_epoch=chosen,parameters=sum(p.numel() for p in model.parameters()),validation=score,wall_s=time.perf_counter()-start,weight_sha256=hashlib.sha256(path.read_bytes()).hexdigest());summary.append(row);save(a.out/'summary.json',summary);print(json.dumps(row),flush=True)

if __name__=='__main__':main()
