#!/usr/bin/env python3
"""Audit saved execution evidence independently of the trial scheduling loop."""
import argparse,hashlib
from navigation_common import *

def main(root):
    root=Path(root);bench=root/'navigation-benchmark-v1';runs=json.loads((bench/'summary.json').read_text())
    frozen=json.loads((bench/'frozen_manifest.json').read_text())
    changed=[k for k,v in frozen['source_sha256'].items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
    assert not changed,changed
    assert len(runs)==24 and len({(r['backend'],r['scene'],r['seed'],r['extra_delay_s']) for r in runs})==24
    audited_plans=0;vb=0.;ab=0.;max_jump={k:0. for k in ['p','v','a']}
    for run in runs:
        folder=bench/f"{run['scene']}_{run['seed']}_{run['backend']}_delay{round(run['extra_delay_s']*1000)}"
        for p in json.loads((folder/'plans.json').read_text()):
            curve=Curve(p);v=float(np.linalg.norm(curve.v.c[:len(p['controls'])-1],axis=1).max());a=float(np.linalg.norm(curve.a.c[:len(p['controls'])-2],axis=1).max())
            assert v<=CONFIG['max_velocity']+1e-6 and a<=CONFIG['max_acceleration']+1e-6,(folder,v,a)
            assert np.linalg.norm(curve.v(curve.duration))<1e-6 and np.linalg.norm(curve.a(curve.duration))<1e-5
            audited_plans+=1;vb=max(vb,v);ab=max(ab,a)
        for e in json.loads((folder/'events.json').read_text()):
            if e['kind']=='commit':assert e['t']<e['start']-CONFIG['deadline_guard']
            if e['kind']=='activate':
                for key in max_jump:max_jump[key]=max(max_jump[key],e['jump'][key])
        for row in (folder/'worker/requests.jsonl').read_text().splitlines():
            request=json.loads(row)['request']
            assert not any(k in request for k in ['history','target_belief','leader','visibility','target_measurement'])
            if 'cloud' in request:
                c=request['cloud'];assert c['stamp']<=request['t']+1e-9 and c['frame']=='world'
    replay=root/'navigation-replay-v1';frames=json.loads((replay/'frame_manifest.json').read_text())
    for b in ['ego','rog']:
        requests=[json.loads(x)['request'] for x in (replay/b/'requests.jsonl').read_text().splitlines()]
        requests=[x for x in requests if x['op']=='update'];assert len(requests)==len(frames)
        for req,expected in zip(requests,frames):
            c=req['cloud'];assert c['origin']==expected['origin'] and c['stamp']==expected['stamp']
            # Archives retain original request metadata; resolve their cloud
            # bytes relative to the replay directory when used on another host.
            data=replay/b/'clouds'/Path(c['binary_xyz_f32']).name
            assert hashlib.sha256(data.read_bytes()).hexdigest()==expected['point_sha256']
    late=json.loads((root/'navigation-deadline-v1/summary.json').read_text())
    assert len(late)==2 and all(x['accepted_plans']==0 and x['rejections'].get('deadline',0)>0 for x in late)
    summary=dict(source_freeze_verified=True,closed_loop_runs=24,successful_runs=sum(r['success'] for r in runs),
        accepted_curves_audited=audited_plans,velocity_control_bound=vb,acceleration_control_bound=ab,
        maximum_handoff_discontinuity=max_jump,matched_replay_frames=len(frames),late_commit_stress_passed=True,
        minimum_obstacle_sample_clearance=min(r['min_sample_clearance_m'] for r in runs if r['scene']!='empty'),
        minimum_ideal_interval_clearance_bound=min(r['conservative_interval_clearance_m'] for r in runs if r['scene']!='empty'))
    save(root/'navigation_audit.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'runs');args=p.parse_args();main(args.root)
