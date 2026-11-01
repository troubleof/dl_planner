"""Actual neural inference plus backend rejection checks, no simulator replacement."""
import argparse
from navigation_common import *
from yopo_frontend import YopoFrontend

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    model=YopoFrontend();sensor=Sensor();scene=make_scene('empty');cloud=sensor.capture(scene,scene['start'],1.)
    d,mask=model.depth(cloud,0.)
    assert 0<mask.mean()<.5 and np.all(d[~mask]==0) and np.max(d)<=.4+1e-6
    q=dict(op='plan',t=1.,start_state=stationary(scene['start']),goal=scene['goal'])
    routes,info=model.propose(cloud,q,a.out,1)
    assert 0<len(routes)<=3 and info['inference_ms']>0
    assert all(np.isfinite(np.asarray(r['points'])).all() for r in routes)
    stale,meta=model.propose(cloud,dict(q,t=1.51),a.out,2);assert stale==[] and meta['reason']=='stale_received_cloud'
    rows=[]
    for backend in ['ego','rog']:
        with Worker(backend,a.out/backend,route_policy='repair') as w:
            for i in range(4):w.request(dict(op='update',t=i*.2,cloud=sensor.capture(scene,scene['start'],i*.2)))
            invalid=dict(id=0,score=0.,points=[[7.,0.,2.5],[8.,0.,2.5]])
            result=w.request(dict(q,neural_routes=[invalid],with_backup=True))
            assert result['ok'] and result['frontend']=='repair' and result['neural_geometric_rejects']==1
            checked=w.request(dict(op='validate',t=1.05,plan=result['plan'],backup_plan=result['backup_plan'],start_state=q['start_state']))
            assert checked['ok']
            late=w.request(dict(op='validate',t=1.3,plan=result['plan'],backup_plan=result['backup_plan'],start_state=q['start_state']))
            assert not late['ok'] and late['reason']=='deadline'
            rows.append(dict(backend=backend,bad_neural_reference_repair_fallback=True,original_validation_passed=True,late_commit_rejected=True))
    save(a.out/'summary.json',dict(model_loaded=True,unobserved_not_free=True,real_inference=True,stale_input_rejected=True,observed_fraction=float(mask.mean()),backend_checks=rows))
    print('YOPO contract checks passed',flush=True)

if __name__=='__main__':main()
