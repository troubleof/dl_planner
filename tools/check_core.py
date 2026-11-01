"""Verify retained artifacts, portable supervision, models and complete cases."""
import argparse, hashlib, json, re, tarfile, tempfile
from pathlib import Path
import numpy as np
import torch
from navigation_common import ROOT, Worker
from recovery_dataset import load_dataset
from recovery_frontend import network_class, probabilities
from extract_case import digest, extract


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    manifest=json.loads((ROOT/'provenance/artifacts.json').read_text())['files']
    for name,record in manifest.items():
        file=ROOT/name;assert file.stat().st_size==record['bytes'] and digest(file)==record['sha256'],name
    data,rows,protocol,corrected=load_dataset(ROOT/'datasets/recovery-v4')
    expected=np.load(ROOT/'survival_models/effective_supervision.npz')
    for key in ['labels','mask']:assert np.array_equal(data[key],expected[key]),key
    base,base_rows,_,_=load_dataset(ROOT/'datasets/recovery-v4',True)
    expected_base=np.load(ROOT/'survival_base_models/effective_supervision.npz')
    for key in ['labels','mask']:assert np.array_equal(base[key],expected_base[key]),key
    maps={split:set() for split in ['train','validation']}
    for row in rows:
        match=re.search(r'_(\d+)_(ego|rog)_',row['source_case']);assert match,row['source_case']
        maps[row['split']].add(int(match[1]))
    assert not maps['train']&maps['validation']
    assert not (maps['train']|maps['validation'])&set(protocol['independent_test_maps'])
    torch.set_num_threads(1);models=[]
    for file in sorted((ROOT/'survival_models').glob('*.pt')):
        checkpoint=torch.load(file,map_location='cpu',weights_only=True)
        model=network_class(checkpoint['kind'])().eval();model.load_state_dict(checkpoint['state_dict'])
        with torch.inference_mode():
            prediction=model(*(torch.from_numpy(data[k][:2]) for k in ['image','previous','state','features'])).numpy()
        assert prediction.shape==(2,45,4) and np.isfinite(prediction).all()
        assert np.all(np.diff(probabilities(prediction),axis=-1)<=1e-7)
        models.append(file.name)
    cases=json.loads((ROOT/'examples/cases/manifest.json').read_text())['cases'];assert len(cases)==5
    references=0;case_reports=[]
    for spec in cases:
        with tarfile.open(ROOT/spec['archive'],'r:gz') as tf:
            members={m.name:m for m in tf.getmembers()};assert set(members)==set(spec['files'])
            for name,hash_value in spec['files'].items():
                assert hashlib.sha256(tf.extractfile(members[name]).read()).hexdigest()==hash_value
            records=[json.loads(line) for line in tf.extractfile('worker/requests.jsonl').read().splitlines()]
            for record in records:
                q=record['request'];cloud=q.get('cloud')
                if cloud:
                    assert cloud['stamp']<=q['t']+1e-8
                    for key in ['binary_xyz_f32','depth_evidence_npz']:
                        if key in cloud:
                            rel=str(Path(cloud[key]).relative_to(spec['original_path']));assert rel in members
                            if key=='binary_xyz_f32':assert members[rel].size==cloud['count']*12
                            references+=1
        case_report=dict(case=spec['id'],files=len(members),original_outcome=spec['outcome'])
        if args.replay:
            # Reconstruct the actual fused maps; this is not a fresh flight or
            # a claim of reproducing old optimizer wall-clock outcomes.
            with tempfile.TemporaryDirectory(prefix='dl-planner-case-') as temp:
                case=Path(temp)/'case';extract(spec,case)
                records=[json.loads(line) for line in (case/'worker/requests.jsonl').read_text().splitlines()]
                escaped=0;updates=0
                with Worker(spec['backend'],Path(temp)/'worker') as worker:
                    for record in records:
                        q=record['request']
                        if 'cloud' in q:worker.request(dict(op='update',t=q['t'],cloud=q['cloud']));updates+=1
                        if q['op']=='escape':
                            request={k:v for k,v in q.items() if k!='cloud'}
                            result=worker.request(request)
                            for stop in result.get('stops',[]):
                                # The worker's escape result contains full hard
                                # validation and a finite grid deadline witness.
                                assert stop['plan']['start']>q['t']
                            escaped+=1
                case_report.update(map_updates_replayed=updates,escape_requests_replayed=escaped)
        case_reports.append(case_report)
    report=dict(passed=True,artifact_files=len(manifest),training_rows=len(rows),base_rows=len(base_rows),training_labels_shape=list(data['labels'].shape),effective_targets_equal_to_frozen_training=True,models=models,train_maps=sorted(maps['train']),validation_maps=sorted(maps['validation']),cloud_references_verified=references,cases=case_reports)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
