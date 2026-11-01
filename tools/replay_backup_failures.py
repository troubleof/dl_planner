"""Frozen-map replay: detect earliest invalid remainder and find a stop witness.
Later original observations are NOT used to claim counterfactual flight safety.
"""
import argparse
from navigation_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for backend in ['ego','rog']:
        name=f'offset_wall_0_{backend}_delay_400_r0';folder=a.baseline/name
        records=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()]
        failure=next(x for x in records if x['response'].get('reason')=='unsafe_execution_prefix')
        plan=failure['request']['executing_plan'];old=Curve(plan)
        events=json.loads((folder/'events.json').read_text());plans=json.loads((folder/'plans.json').read_text())
        commit=next(e for p,e in zip(plans,[e for e in events if e['kind']=='commit']) if digest(p)==digest(plan))
        detected=None
        with Worker(backend,a.out/name,route_policy='repair') as w:
            for x in records:
                req=x['request']
                if 'cloud' not in req:continue
                cloud=dict(req['cloud'])
                if not Path(cloud['binary_xyz_f32']).exists():cloud['binary_xyz_f32']=str(folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name)
                q=dict(op='update',t=req['t'],cloud=cloud)
                if req['t']>=max(commit['t'],plan['start'])-1e-9:
                    q.update(op='guard',execution_schedule=[plan],nominal_plan=plan,find_stop=True)
                r=w.request(q)
                if r.get('nominal_invalid_intervals'):
                    detected=dict(backend=backend,folder=name,detected_at=req['t'],scan_stamp=cloud['stamp'],old_failure_at=failure['request']['t'],
                        earlier_detection_s=failure['request']['t']-req['t'],report=r)
                    if r.get('stop_found'):
                        backup=Curve(r['stop_plan']);before=old.state(backup.start);after=backup.state(backup.start)
                        jump={k:float(np.linalg.norm(np.asarray(before[k])-after[k])) for k in ['p','v','a']}
                        ready=round_tick(req['t']+r['rpc_ms']/1000+.08+CONFIG['publication_delay']+CONFIG['dt'])
                        stop_state=backup.state(backup.start+backup.duration)
                        detected.update(jump=jump,ready=ready,deadline=r['verified_stop_commit_deadline'],within_budget=ready<r['verified_stop_commit_deadline'],
                            stop_speed=float(np.linalg.norm(stop_state['v'])),stop_acceleration=float(np.linalg.norm(stop_state['a'])))
                    break
        assert detected,'historical invalid remainder not reproduced'
        rows.append(detected);print(json.dumps(detected),flush=True)
    save(a.out/'summary.json',rows)

if __name__=='__main__':main()
