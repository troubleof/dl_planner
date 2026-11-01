import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from navigation_common import Curve,save
from direct_frontend import net_class,decode,controls_torch,rotation

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);a=ap.parse_args();out=a.models/'deploy';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);d=np.load(a.dataset/'dataset.npz');rows=json.loads((a.dataset/'records.json').read_text());results=[]
    x=torch.from_numpy(d['image']);s=torch.from_numpy(d['state'])
    for seed in [11,22,33]:
        model=net_class()().eval();model.load_state_dict(torch.load(a.models/f'direct_s{seed}.pt',map_location='cpu',weights_only=True)['state_dict'])
        exported=torch.jit.trace(model,(x[:1],s[:1]));path=out/f'direct_s{seed}.torchscript';exported.save(str(path));loaded=torch.jit.load(str(path))
        maximum=0.;jumps={k:0. for k in ['p','v','a']};decoder_error=0.
        for i,row in enumerate(rows):
            with torch.inference_mode():raw,pred=model(x[i:i+1],s[i:i+1]);r2,p2=loaded(x[i:i+1],s[i:i+1]);qtorch,_=controls_torch(raw,s[i:i+1])
            maximum=max(maximum,float((raw-r2).abs().max()),float((pred-p2).abs().max()))
            plans,_=decode(row['query'],raw.numpy()[0],pred.numpy()[0]);start=np.asarray(row['query']['start_state']['p']);rot,_=rotation(row['query']['goal'],start)
            for candidate in plans:
                c=Curve(candidate['plan']);state=c.state(c.start)
                for key in jumps:jumps[key]=max(jumps[key],float(np.linalg.norm(np.asarray(state[key])-row['query']['start_state'][key])))
                assert np.linalg.norm(c.v(c.duration))<1e-6 and np.linalg.norm(c.a(c.duration))<1e-5
                local=(np.asarray(candidate['plan']['controls'])-start)@rot
                decoder_error=max(decoder_error,float(np.max(np.abs(local-qtorch.numpy()[0,candidate['id']]))))
        assert maximum<1e-6 and max(jumps.values())<1e-6 and decoder_error<5e-6
        results.append(dict(seed=seed,artifact=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),observations=len(rows),
            max_export_error=maximum,max_start_pva_error=jumps,max_torch_numpy_decoder_error=decoder_error,parameters=sum(p.numel() for p in model.parameters())))
    save(out/'verification.json',dict(passed=True,models=results,interface='TorchScript predicts timed parameters and scores; exact-PVA decoder and original validator/stop pipeline remain necessary'))
    print(json.dumps(dict(passed=True,models=results)),flush=True)

if __name__=='__main__':main()
