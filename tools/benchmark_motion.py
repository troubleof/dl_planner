"""CPU single-thread warm inference/feature latency; no fabricated board timing."""
import argparse,json,platform,time,hashlib
from pathlib import Path
import numpy as np
import torch
from navigation_common import ROOT,save
from motion_frontend import MotionFrontend,net_class

def stats(v):return {**{f'p{p}':float(np.percentile(v,p)) for p in [50,95,99]},'max':float(np.max(v))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--repeats',type=int,default=500);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);rows=json.loads((a.dataset/'records.json').read_text());picked=[r for r in rows if r['split']=='validation' and r['motion_variant']=='original'][::5];inputs=[]
    for r in picked:
        cloud=dict(r['cloud']);cloud['points']=np.fromfile(cloud['binary_xyz_f32'],dtype='<f4').reshape(-1,3)
        if 'depth_metadata_json' in cloud:cloud.update(json.loads(Path(cloud['depth_metadata_json']).read_text()))
        inputs.append((cloud,r['query'],r['backend']))
    results=[]
    for variant in ['control','features','labels','both']:
        path=ROOT/'motion_models'/f'{variant}_s11.pt';model=MotionFrontend('ego',path);measure={k:[] for k in ['preprocess_ms','inference_ms','decode_ms','compute_ms']}
        for i in range(a.repeats+50):
            cloud,q,backend=inputs[i%len(inputs)];model.backend=backend;_,_,timing=model.compute(cloud,q)
            if i>=50:
                for key in measure:measure[key].append(timing[key])
        # Export fixed-input-shape CPU artifact; verify multiple real states.
        channels=model.channels;traced=torch.jit.trace(model.model,(torch.zeros(1,*model.input_shape),torch.zeros(1,32)),check_trace=True)
        error=0.;jit=[]
        from motion_frontend import observation
        for cloud,q,backend in inputs:
            im,st=observation(cloud,q,backend,channels);tensors=(torch.from_numpy(im[None]),torch.from_numpy(st[None]))
            with torch.inference_mode():
                expected=model.model(*tensors);actual=traced(*tensors)
                error=max(error,max(float((x-y).abs().max()) for x,y in zip(expected,actual)))
        assert error<1e-6
        for _ in range(50):traced(*tensors)
        with torch.inference_mode():
            for _ in range(a.repeats):t=time.perf_counter();traced(*tensors);jit.append((time.perf_counter()-t)*1000)
        target=a.out/f'{variant}_s11.torchscript';torch.jit.save(traced,str(target))
        results.append(dict(variant=variant,parameters=model.metadata['parameters'],channels=channels,eager={k:stats(v) for k,v in measure.items()},torchscript_ms=stats(jit),export_max_error=error,export_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    save(a.out/'benchmark.json',dict(platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],torch=torch.__version__,threads=1,repeats=a.repeats,models=results,
        scope='warm CPU feature+inference+decode; map, transport, evidence I/O measured separately in closed-loop RPC; not onboard or WCET claim'))
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
