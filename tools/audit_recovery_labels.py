"""Independently verify causal restoration, identical observations and label arithmetic."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from navigation_common import save
from branch_resume import equivalent
from collect_recovery_branches import measure
from recovery_frontend import CandidateBank


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);a=ap.parse_args()
    records=json.loads((a.dataset/'records.json').read_text());data=np.load(a.dataset/'dataset.npz')
    generators={};source_cache={};branches=0;failure_counts={};state_comparisons=[]
    for i,record in enumerate(records):
        source=Path(record['source_folder'])
        if not source.exists():source=a.dataset.parent/'source_prefixes'/source.name
        source_file=source/'worker/requests.jsonl'
        assert hashlib.sha256(source_file.read_bytes()).hexdigest()==record['source_sha256']
        if str(source) not in source_cache:source_cache[str(source)]=[json.loads(x) for x in source_file.read_text().splitlines()]
        original=source_cache[str(source)];index=record['source_request_index'];target=original[index]['request']
        baseline_initial=None;baseline_bank=None
        for item in record['branches']:
            folder=a.dataset/item['folder'];result=json.loads((folder/'result.json').read_text())
            assert item['result']==result
            long,short,diagnostic=measure(folder,result,np.asarray(target['goal']))
            assert np.max(np.abs(np.asarray(long)-data['long'][i,item['action']]))<1e-6
            assert np.max(np.abs(np.asarray(short)-data['short'][i,item['action']]))<1e-6
            log=[json.loads(x) for x in (folder/'worker/requests.jsonl').read_text().splitlines()]
            initial=diagnostic['initial_request'];bank=initial['neural_input']['bank']
            for k in ['p','v','a']:
                assert np.allclose(initial['start_state'][k],target['start_state'][k],atol=1e-8,rtol=0)
            assert initial['t']==target['t'] and initial['neural_input']['action']==item['action']
            assert initial['neural_routes']==bank[item['action']]
            restored=[r for r in log if r['request'].get('_skip_frontend',False)]
            expected=[r for r in original[:index] if r['request']['op']=='plan']
            assert len(restored)==len(expected)
            assert all(equivalent(x['response'],y['response']) for x,y in zip(restored,expected))
            traj=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1)
            if baseline_initial is None:baseline_initial=traj[0,:10];baseline_bank=bank
            else:
                assert np.allclose(traj[0,:10],baseline_initial,rtol=0,atol=1e-8)
                assert bank==baseline_bank
            for frame in json.loads((folder/'sensor_frames.json').read_text()):
                if frame['stamp']<=target['t']+1e-9:continue
                point=traj[np.argmin(np.abs(traj[:,0]-frame['stamp']))]
                assert abs(point[0]-frame['stamp'])<1e-8
                assert np.max(np.abs(point[1:4]-frame['origin']))<1e-7
            # Regenerate the deployed inputs and actual candidate bundles from received clouds.
            latest=None
            for entry in log:
                if 'cloud' in entry['request']:latest=entry['request']['cloud']
                if entry['request']==initial:break
            assert latest is not None and latest['stamp']<=initial['t']
            cloud=dict(latest);cloud['points']=np.fromfile(folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name,dtype='<f4').reshape(-1,3).tolist()
            if record['backend'] not in generators:generators[record['backend']]=CandidateBank(record['backend'])
            image,state,feature,rebuilt=generators[record['backend']].build(cloud,initial)
            assert np.array_equal(image,data['image'][i]) and np.array_equal(state,data['state'][i])
            assert np.array_equal(feature,data['candidates'][i]) and rebuilt==bank
            branches+=1;failure_counts[result['outcome']]=failure_counts.get(result['outcome'],0)+1
        state_comparisons.append(dict(state=i,split=record['split'],hard=record['hard'],
            actual_short=data['short'][i].tolist(),actual_long=data['long'][i].tolist()))
    summary=dict(verified=True,states=len(records),branches=branches,outcomes=failure_counts,
        source_hashes_verified=True,restored_plans_verified=True,initial_pva_and_banks_equal=True,
        actual_candidates_recomputed=True,post_fork_clouds_follow_branch=True,label_arithmetic_verified=True,
        positive_failures=int(data['long'][...,1].sum()),positive_stalls=int(data['long'][...,2].sum()),paired=state_comparisons)
    save(a.dataset/'audit.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='paired'}),flush=True)


if __name__=='__main__':main()
