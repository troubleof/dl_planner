"""Export CPU TorchScript and check all saved train/validation observations."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from navigation_common import save
from trajectory_frontend import model_class,FORMAT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    data=np.load(a.dataset/'dataset.npz');image=torch.from_numpy(data['image']);state=torch.from_numpy(data['state'])
    results=[]
    for seed in [11,22,33]:
        weight=a.models/f'trajectory_s{seed}.pt';ckpt=torch.load(weight,map_location='cpu',weights_only=True)
        assert ckpt['format']==FORMAT
        model=model_class()().eval();model.load_state_dict(ckpt['state_dict'])
        with torch.inference_mode():
            traced=torch.jit.trace(model,(image[:1],state[:1]))
            path=a.out/f'trajectory_s{seed}.torchscript';traced.save(str(path))
            loaded=torch.jit.load(str(path)).eval();errors=np.zeros(3);rank_differences=0
            for offset in range(0,len(image),32):
                batch=image[offset:offset+32],state[offset:offset+32]
                reference=model(*batch);actual=loaded(*batch)
                errors=np.maximum(errors,[float(torch.max(torch.abs(x-y))) for x,y in zip(reference,actual)])
                def ranking(v):return torch.argsort(v[0]-2*(1-torch.sigmoid(v[1])),dim=1,descending=True)
                rank_differences+=int((ranking(reference)!=ranking(actual)).any(dim=1).sum())
            assert errors.max()<1e-6 and rank_differences==0
            times=[]
            for i in range(220):
                begin=time.perf_counter();loaded(image[:1],state[:1]);ms=(time.perf_counter()-begin)*1000
                if i>=20:times.append(ms)
        results.append(dict(seed=seed,observations_checked=len(image),max_abs_error=errors.tolist(),
            ranking_differences=rank_differences,artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),bytes=path.stat().st_size,
            single_thread_forward_ms=dict(p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95))),
            timing_scope='CPU forward only, not full navigation latency'))
    save(a.out/'verification.json',dict(format=FORMAT,passed=True,models=results));print(json.dumps(results),flush=True)


if __name__=='__main__':main()
