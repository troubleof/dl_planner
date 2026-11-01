#!/usr/bin/env python3
"""Audit the saved routing ablation, including accepted curves and commit records."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from navigation_common import ROOT,Curve,CONFIG,save,digest

def main(root):
    root=Path(root);rows=json.loads((root/'summary.json').read_text());frozen=json.loads((root/'frozen_manifest.json').read_text())
    changed=[k for k,v in frozen['source_sha256'].items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
    assert not changed,changed
    assert len(rows)==frozen['expected_closed_loop_runs']
    assert len({(r['scene'],r['seed'],r['backend'],r['route_policy'],r['extra_delay_s'],r['repeat']) for r in rows})==len(rows)
    common=None;scenes={};accepted=0;max_v=0.;max_a=0.;max_jumps={k:0. for k in ['p','v','a']};grouped={}
    for r in rows:
        folder=root/r['folder'];cfg=json.loads((folder/'config.json').read_text());key=(r['scene'],r['seed'])
        c={k:v for k,v in cfg.items() if k not in ['route_policy','backend','extra_delay_s']}
        if common is None:common=c
        assert c==common,'non-routing configuration changed'
        scenes.setdefault(key,r['scene_sha256']);assert scenes[key]==r['scene_sha256']
        trajectory=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1)
        assert np.isfinite(trajectory).all()
        if r['success']:assert trajectory[:,-1].min()>=0
        events=json.loads((folder/'events.json').read_text());plans=json.loads((folder/'plans.json').read_text())
        commits=[e for e in events if e['kind']=='commit'];assert len(plans)==len(commits)==r['accepted_plans']
        traffic=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()]
        validations={digest(x['request']['plan']):x for x in traffic if x['request']['op']=='validate' and x['response']['ok']}
        for plan,event in zip(plans,commits):
            assert event['t']<plan['start']-CONFIG['deadline_guard']
            checked=validations[digest(plan)]
            assert checked['response']['map_version']==event['map_version']
            assert checked['response']['map_stamp']==event['map_stamp']
            curve=Curve(plan);n=len(plan['controls'])
            v=float(np.linalg.norm(curve.v.c[:n-1],axis=1).max());a=float(np.linalg.norm(curve.a.c[:n-2],axis=1).max())
            assert v<=CONFIG['max_velocity']+1e-6 and a<=CONFIG['max_acceleration']+1e-6
            assert np.linalg.norm(curve.v(curve.duration))<1e-6 and np.linalg.norm(curve.a(curve.duration))<1e-5
            accepted+=1;max_v=max(max_v,v);max_a=max(max_a,a)
        for e in events:
            if e['kind']=='activate':
                for k in max_jumps:max_jumps[k]=max(max_jumps[k],e['jump'][k])
        for x in traffic:
            req,resp=x['request'],x['response']
            if 'cloud' in req:assert req['cloud']['stamp']<=req['t']+1e-9
            if req['op']=='plan' and resp['ok']:
                assert abs(resp['plan']['start']-req['t']-CONFIG['handoff'])<1e-8
                assert resp['route_map_version']==resp['map_version']
        plans_timing=[x['response'] for x in traffic if x['request']['op']=='plan']
        group=grouped.setdefault((r['backend'],r['route_policy']),dict(runs=[],timings=[],committed_sources=Counter()))
        group['runs'].append(r);group['timings']+=plans_timing;group['committed_sources'].update(e['route_source'] for e in commits)
    groups=[]
    for (backend,policy),g in grouped.items():
        rr=g['runs'];tt=g['timings']
        def pct(key):return dict(zip(['p50','p95','p99'],map(float,np.percentile([x[key] for x in tt if key in x],[50,95,99]))))
        groups.append(dict(backend=backend,policy=policy,runs=len(rr),successes=sum(x['success'] for x in rr),
            full_astar_calls=sum(x['full_astar_calls'] for x in rr),local_astar_calls=sum(x['local_astar_calls'] for x in rr),
            repaired_routes=sum(x['repaired_routes'] for x in rr),reused_routes=sum(x['reused_routes'] for x in rr),committed_sources=dict(g['committed_sources']),
            search_ms=pct('search_ms'),astar_ms=pct('astar_ms'),planning_rpc_ms=pct('rpc_ms'),
            mean_flight_seconds=float(np.mean([x['duration_s'] for x in rr])),mean_path_m=float(np.mean([x['path_length_m'] for x in rr])),
            minimum_sample_clearance_m=min(x['min_sample_clearance_m'] for x in rr if x['scene']!='empty')))
    report=dict(source_freeze_verified=True,common_configuration_verified=True,closed_loop_runs=len(rows),
        successes=sum(r['success'] for r in rows),failures=[dict(folder=r['folder'],outcome=r['outcome']) for r in rows if not r['success']],
        accepted_curves_audited=accepted,velocity_control_bound=max_v,acceleration_control_bound=max_a,maximum_handoff_discontinuity=max_jumps,
        minimum_ideal_interval_clearance_bound=min(r['conservative_interval_clearance_m'] for r in rows if r['scene']!='empty'),groups=groups)
    save(root/'audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
