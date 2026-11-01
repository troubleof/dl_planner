import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from navigation_common import save
from recovery_frontend import net_class,utilities

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    parts=[np.load(p/'dataset.npz') for p in a.datasets]
    x,s,c=[torch.from_numpy(np.concatenate([p[k] for p in parts])) for k in ['image','state','candidates']]
    rows=[]
    for weight in sorted(a.models.glob('*.pt')):
        ckpt=torch.load(weight,map_location='cpu',weights_only=True);model=net_class()().eval();model.load_state_dict(ckpt['state_dict'])
        with torch.inference_mode():
            module=torch.jit.trace(model,(x[:1],s[:1],c[:1]));path=a.out/(weight.stem+'.torchscript');module.save(str(path))
            loaded=torch.jit.load(str(path)).eval();actual=loaded(x,s,c);expected=model(x,s,c)
            error=float((actual-expected).abs().max());assert error<1e-6
            assert torch.equal(utilities(actual).argsort(dim=1),utilities(expected).argsort(dim=1))
        rows.append(dict(weight=weight.name,artifact=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),observations=len(x),max_error=error,
            trainable_scorer_parameters=sum(p.numel() for p in model.parameters()),frozen_generator_parameters=2*56751,total_parameters=2*56751+sum(p.numel() for p in model.parameters())))
    save(a.out/'verification.json',dict(passed=True,models=rows,scope='Scorer export only; the two unchanged candidate generators and input/decoder are also required.'))
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
