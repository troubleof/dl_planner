"""Independent timing/dynamics and executed-sample/map replay audit."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,help='Extracted original_failures, for relocated prefix-cloud replay')
    a=ap.parse_args()
    sys.path.insert(0,str(a.root/'tools'))
    from navigation_common import Worker,Curve,CONFIG,save
    a.out.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((a.run/'frozen_manifest.json').read_text())
    assert all(hashlib.sha256((a.root/p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items())
    results=[];bounds=[0.,0.];jumps={k:0. for k in ['p','v','a']};num_curves=0;num_clouds=0;neural_calls=0;model_outputs=0;models={}
    for row in json.loads((a.run/'summary.json').read_text()):
        folder=a.run/row['folder'];events=json.loads((folder/'events.json').read_text());bundles=json.loads((folder/'plans.json').read_text())
        frames=json.loads((folder/'sensor_frames.json').read_text());traj=np.loadtxt(folder/'trajectory.csv',delimiter=',',skiprows=1)
        original=[json.loads(s) for s in (folder/'worker/requests.jsonl').read_text().splitlines()]
        last_cloud=None;seen_clouds=[]
        for i,record in enumerate(original,1):
            q,r=record['request'],record['response']
            if 'cloud' in q:
                last_cloud=q['cloud'];seen_clouds.append(last_cloud)
            if q.get('neural_input',{}).get('reason')=='ok':
                neural_calls+=1;ni=q['neural_input']
                assert last_cloud is not None and ni['capture_stamp']==last_cloud['stamp']<=q['t']
                assert ni['frontend_ms']<=r['rpc_ms']+1e-8
                data=np.load(folder/'worker/neural'/f'{i:05d}.npz')
                if 'rays' in data:
                    from context_frontend import features
                    def loaded(c):
                        cloud=dict(c);cloud['points']=np.fromfile(folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name,dtype='<f4').reshape(-1,3).tolist();return cloud
                    previous=loaded(seen_clouds[-2]) if len(seen_clouds)>1 else None
                    rays,local_map,state=features(loaded(last_cloud),previous,q)
                    assert np.array_equal(data['rays'],rays) and np.array_equal(data['local_map'],local_map) and np.array_equal(data['state'],state)
                    assert q['neural_input']['previous_stamp']<=q['neural_input']['capture_stamp']<=q['t']
                    assert r['rpc_ms']>=q['context_rpc_ms']+ni['frontend_ms']-1e-6
                    assert original[i-2]['request']['op']=='context'
                    assert original[i-2]['response']['context']==q['map_context']
                    if len(data['prediction']):
                        import torch
                        from context_frontend import net_class,predicted_utility,ACTION_IDS
                        torch.set_num_threads(1)
                        if row['policy'] not in models:
                            loaded_models=[]
                            for spec in json.loads((folder/'model.json').read_text())['weights']:
                                path=Path(spec['path'])
                                if not path.exists():path=a.root/'context_models'/path.parent.name/path.name
                                assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
                                checkpoint=torch.load(path,map_location='cpu',weights_only=True);model=net_class()().eval();model.load_state_dict(checkpoint['state_dict']);loaded_models.append(model)
                            models[row['policy']]=loaded_models
                        with torch.inference_mode():
                            prediction=torch.stack([model(torch.from_numpy(rays[None]),torch.from_numpy(local_map[None]),torch.from_numpy(state[None]))[0] for model in models[row['policy']]])
                            assert np.max(np.abs(prediction.numpy()-data['prediction']))<1e-5
                            values=predicted_utility(prediction);mean=values.mean(dim=0).numpy();spread=values.std(dim=0,unbiased=False).numpy()
                        assert np.max(np.abs(mean-ni['utility']))<1e-5 and np.max(np.abs(spread-ni['disagreement']))<1e-5
                        chosen=int((mean-spread).argmax())
                        if mean[chosen]-spread[chosen]<mean[4]+spread[4]+.05:chosen=4
                        if np.linalg.norm(np.asarray(q['goal'])-q['start_state']['p'])<3.:chosen=4
                        assert chosen==ni['action'] and ACTION_IDS[chosen]==ni['candidate_id']
                        assert ni['routes']==q['neural_routes'];model_outputs+=1
                elif 'candidates' in data:
                    import torch
                    from recovery_frontend import CandidateBank,net_class,choose
                    spec=json.loads((folder/'model.json').read_text())
                    key='bank_'+row['backend']
                    if key not in models:models[key]=CandidateBank(row['backend'])
                    assert [(v['seed'],v['sha256']) for v in models[key].weights]==[(v['seed'],v['sha256']) for v in spec['generators']]
                    cloud=dict(last_cloud)
                    cloud['points']=np.fromfile(folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name,dtype='<f4').reshape(-1,3).tolist()
                    expected_image,expected_state,features,bank=models[key].build(cloud,q)
                    assert np.array_equal(expected_image,data['range_image'])
                    assert np.array_equal(expected_state,data['state'])
                    assert np.array_equal(features,data['candidates'])
                    if spec['mode']=='bank_fixed':action=1
                    else:
                        key='scorer_'+row['policy']
                        if key not in models:
                            path=a.root/'recovery_models'/Path(spec['weight']).name
                            assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
                            ckpt=torch.load(path,map_location='cpu',weights_only=True)
                            model=net_class()().eval();model.load_state_dict(ckpt['state_dict']);models[key]=model
                        with torch.inference_mode():
                            pred=models[key](torch.from_numpy(expected_image[None]),torch.from_numpy(expected_state[None]),torch.from_numpy(features[None]))
                        assert np.max(np.abs(pred.numpy()[0]-data['prediction']))<1e-6
                        action,value=choose(pred,bank)
                        assert np.max(np.abs(value-ni['utility']))<1e-5
                        model_outputs+=1
                    assert bank==ni['bank'] and action==ni['action']
                    assert bank[action]==q['neural_routes']==ni['routes']
                elif 'depth' in data:
                    assert np.all(data['depth'][~data['observed_mask']]==0)
                else:
                    from adapted_frontends import inputs
                    cloud=dict(last_cloud)
                    cloud['points']=np.fromfile(folder/'worker/clouds'/Path(cloud['binary_xyz_f32']).name,dtype='<f4').reshape(-1,3).tolist()
                    expected_image,expected_state=inputs(cloud,q)
                    assert np.array_equal(data['range_image'],expected_image)
                    assert np.array_equal(data['state'],expected_state)
                    if 'offsets' in data:
                        import torch
                        from trajectory_frontend import model_class,decode,FORMAT
                        torch.set_num_threads(1)
                        spec=json.loads((folder/'model.json').read_text())
                        if row['policy'] not in models:
                            path=a.root/'trajectory_models'/Path(spec['weight']).name
                            assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
                            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                            assert checkpoint['format']==FORMAT
                            model=model_class()().eval();model.load_state_dict(checkpoint['state_dict'])
                            models[row['policy']]=model
                        with torch.inference_mode():
                            score,logit,offset=models[row['policy']](torch.from_numpy(expected_image[None]),torch.from_numpy(expected_state[None]))
                        score=score.numpy()[0];feasible=torch.sigmoid(logit).numpy()[0];offset=offset.numpy()[0]
                        assert np.max(np.abs(score-data['scores']))<1e-6
                        assert np.max(np.abs(feasible-data['feasible']))<1e-6
                        assert np.max(np.abs(offset-data['offsets']))<1e-6
                        routes,_=decode(q,score,feasible,offset,spec['fixed_geometry'])
                        assert routes==q['neural_routes'],'decoded geometry does not match sent proposals'
                        model_outputs+=1
                if ni['capture_stamp']>=traj[0,0]:
                    captured=traj[np.argmin(np.abs(traj[:,0]-ni['capture_stamp']))]
                    assert abs(captured[0]-ni['capture_stamp'])<1e-8
                    st=last_cloud['capture_state']
                    assert np.max(np.abs(np.asarray(st['p']+st['v']+st['a'])-captured[1:10]))<1e-7
        prefix=folder/'prefix_initialization.json';pre_init=json.loads(prefix.read_text()) if prefix.exists() else None
        for pair in bundles:
            nominal=Curve(pair['nominal']);backup=Curve(pair['backup'])
            for curve in [nominal,backup]:
                vb=float(np.linalg.norm(curve.v.c,axis=1).max());ab=float(np.linalg.norm(curve.a.c,axis=1).max())
                assert vb<=CONFIG['max_velocity']+1e-6 and ab<=CONFIG['max_acceleration']+1e-6
                assert np.linalg.norm(curve.v(curve.duration))<1e-6 and np.linalg.norm(curve.a(curve.duration))<1e-5
                bounds=[max(bounds[0],vb),max(bounds[1],ab)];num_curves+=1
            before=nominal.state(backup.start);after=backup.state(backup.start)
            for k,tol in [('p',1e-6),('v',1e-6),('a',1e-5)]:assert np.linalg.norm(np.asarray(before[k])-after[k])<tol
        for event in events:
            if event['kind']=='activate':
                for k,tol in [('p',1e-6),('v',1e-6),('a',1e-5)]:
                    jumps[k]=max(jumps[k],event['jump'][k]);assert event['jump'][k]<tol
            if event['kind']=='commit':
                assert event['t']<event['start']-CONFIG['deadline_guard']-1e-10
                delivered=[f['stamp'] for f in frames if f['delivered_at'] is not None and f['delivered_at']<=event['t']+1e-9 and not f['dropped']]
                assert not delivered or max(delivered)<=event['map_stamp']+1e-9,'newer arrived scan ignored'
        updates=[r['request'] for r in original if 'cloud' in r['request']]
        bad=[];samples=0;cloud_seen=set()
        from context_frontend import ContextFrontend
        with Worker(row['backend'],a.out/row['folder'],route_policy='repair',frontend=ContextFrontend(mode='repair')) as w:
            for i,q in enumerate(updates):
                cloud=q['cloud'];num_clouds+=1
                if 'binary_xyz_f32' in cloud and not Path(cloud['binary_xyz_f32']).exists():
                    file=Path(cloud['binary_xyz_f32'])
                    if pre_init and Path(pre_init['prefix_source']).name in file.parts:
                        assert a.baseline is not None,'pass --baseline to relocate historical prefix clouds'
                        file=a.baseline/Path(pre_init['prefix_source']).name/'worker/clouds'/file.name
                    else:file=folder/'worker/clouds'/file.name
                    assert file.exists(),file
                    cloud['binary_xyz_f32']=str(file.resolve())
                assert cloud['stamp'] not in cloud_seen;cloud_seen.add(cloud['stamp'])
                if 'sequence' in cloud:
                    f=frames[cloud['sequence']]
                    assert not f['dropped'] and f['delivered_at'] is not None
                    assert f['stamp']<=f['scheduled_arrival']<=f['delivered_at']+1e-9<=q['t']+1e-8
                    assert np.max(np.abs(np.asarray(cloud['origin'])-f['origin']))<1e-9
                    assert abs(cloud['stamp']-f['stamp'])<1e-9
                w.request(dict(op='update',t=q['t'],cloud=cloud))
                # Replay each original read-only local-map query in its exact
                # fusion interval, including the per-voxel observation times.
                source_index=next(k for k,x in enumerate(original) if x['request'] is q)
                next_source=next((k for k in range(source_index+1,len(original)) if 'cloud' in original[k]['request']),len(original))
                for item in original[source_index:next_source]:
                    cq,cr=item['request'],item['response']
                    if cq['op']=='context' and 'context' in cr:
                        replay=w.request({k:cq[k] for k in ['op','t','start_state','goal']})
                        assert np.max(np.abs(np.asarray(replay['context']['grid'])-cr['context']['grid']))<1e-12
                until=updates[i+1]['t'] if i+1<len(updates) else traj[-1,0]+.01
                subset=traj[(traj[:,0]>=q['t']-1e-9)&(traj[:,0]<until-1e-9)]
                if len(subset):
                    cells=w.request(dict(op='query',t=q['t'],points=subset[:,1:4].tolist()))['cells'];samples+=len(cells)
                    bad.extend([dict(t=float(x[0]),state=cell,point=x[1:4].tolist()) for x,cell in zip(subset,cells) if cell[0]!=0 or cell[1]])
        records=dict(folder=row['folder'],outcome=row['outcome'],executed_samples_checked=samples,invalid_executed_samples=bad,
            final_speed=row['final_speed'],backup_activations=row['backup_activations'],
            backup_activated_while_busy=sum(e['kind']=='activate' and e['curve_kind']=='backup' and e.get('pipeline_busy',False) for e in events),
            nominal_invalid_reports=sum(bool(e['report'].get('nominal_invalid_intervals')) for e in events if e['kind']=='execution_check'))
        # Preserve any violation and continue auditing the remaining failures.
        results.append(records);save(a.out/'progress.json',results);print(json.dumps(records),flush=True)
    summary=dict(frozen_source_verified=True,cases=len(results),curves_audited=num_curves,cloud_inputs_audited=num_clouds,
        max_velocity_control_bound=bounds[0],max_acceleration_control_bound=bounds[1],maximum_handoff_jump=jumps,
        executed_map_invalid_samples=sum(len(r['invalid_executed_samples']) for r in results),neural_inputs_audited=neural_calls,model_outputs_recomputed=model_outputs,cases_detail=results,
        limitations=['Independent map replay checks executed samples at 10 ms, not continuous swept volume.',
                    'Failure episodes end at reported loss of validated escape; no safe stop is inferred for them.',
                    'Latest arrived cloud gate checked only at new bundle commits. Previously committed stops execute without fresh admission.'])
    save(a.out/'audit.json',summary)

if __name__=='__main__':main()
