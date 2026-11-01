"""Replay frozen failed episodes without changing their source or evidence."""
import argparse,json
from pathlib import Path
from navigation_common import Worker,Curve,save,round_tick,CONFIG

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for run in a.runs:
        for item in json.loads((run/'summary.json').read_text()):
            if item['outcome']!='no_validated_escape':continue
            folder=run/item['folder'];records=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()]
            events=json.loads((folder/'events.json').read_text());detected=next(e for e in events if e['kind']=='execution_check' and not e['report']['ok'])
            failed=next((i for i,x in enumerate(records) if not x['response'].get('execution_guard',x['response'] if x['request']['op']=='guard' else {'ok':True})['ok']),None)
            assert failed is not None
            q=records[failed]['request'];guard=records[failed]['response'].get('execution_guard',records[failed]['response']);t=q['t'];bad=guard['first_invalid_time'];at=bad+.006
            curve=Curve(q['execution_schedule'][0])
            for pl in q['execution_schedule'][1:]:
                if pl['start']<=at:curve=Curve(pl)
            point=curve.state(at)['p'];name=run.name+'__'+item['folder']
            with Worker(item['backend'],a.out/name) as w:
                for x in records[:failed]:
                    if 'cloud' in x['request']:w.request(dict(op='update',t=x['request']['t'],cloud=x['request']['cloud']))
                before=w.request(dict(op='query',t=t,points=[point]))['cells'][0]
                if 'cloud' in q:w.request(dict(op='update',t=t,cloud=q['cloud']))
                after=w.request(dict(op='query',t=t,points=[point]))['cells'][0]
                r=w.request(dict(op='escape',t=detected['t'],execution_schedule=q['execution_schedule']))
                ready=round_tick(detected['t']+r['rpc_ms']/1000+CONFIG['publication_delay']+CONFIG['dt'])
                usable=[s for s in r['stops'] if ready<s['commit_deadline'] and (bad is None or ready<bad)]
            row=dict(case=name,split=item['split'],policy=item['policy'],request_t=t,detected_t=detected['t'],first_invalid_t=bad,lead_s=bad-detected['t'],before_cell=before,after_cell=after,probe_point=point,search_rpc_ms=r['rpc_ms'],ready=ready,feasible_stops=len(r['stops']),usable_stops=len(usable),report=r)
            rows.append(row);save(a.out/'diagnostic.json',rows);print(json.dumps({k:row[k] for k in ['case','lead_s','before_cell','after_cell','usable_stops']}),flush=True)

if __name__=='__main__':main()
