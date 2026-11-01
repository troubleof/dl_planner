import argparse,json
from pathlib import Path
import numpy as np
import torch
from navigation_common import ROOT,save,Sensor,make_scene,stationary
from recovery_frontend import CandidateBank,RecoveryFrontend,choose
from trajectory_frontend import TrajectoryFrontend


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    scene=make_scene('forest',3202);cloud=Sensor().capture(scene,scene['start'],.8)
    q=dict(op='plan',t=1.,goal=scene['goal'],start_state=stationary(scene['start']),with_backup=True)
    bank=CandidateBank('ego');image,state,features,routes=bank.build(cloud,q)
    assert image.shape==(3,17,72) and state.shape==(23,) and features.shape==(4,37)
    for action,(seed,fixed) in enumerate([(11,True),(11,False),(33,False)],1):
        old=TrajectoryFrontend(ROOT/'trajectory_models'/f'trajectory_s{seed}.pt',fixed=fixed)
        expected,_=old.propose(cloud,q,a.out/f'old{action}',1)
        assert routes[action]==expected,'candidate bank changed frozen generator behavior'
    # The shared scorer's action dimension is permutation equivariant.
    for target,seed in [('short',11),('long',11),('long',22),('long',33)]:
        f=RecoveryFrontend('ego',ROOT/'recovery_models'/f'{target}_s{seed}.pt')
        with torch.inference_mode():
            base=f.model(torch.from_numpy(image[None]),torch.from_numpy(state[None]),torch.from_numpy(features[None]))
            perm=[2,0,3,1]
            shuffled=f.model(torch.from_numpy(image[None]),torch.from_numpy(state[None]),torch.from_numpy(features[perm][None]))
        assert torch.max(torch.abs(base[:,perm]-shuffled))<1e-6
        chosen,info=f.propose(cloud,q,a.out/f'{target}{seed}',1)
        assert chosen==routes[info['action']]
        for bad,request in [(None,q),(cloud,dict(q,t=1.31)),(cloud,dict(q,t=.7)),(dict(cloud,points=cloud['points'][:-1]),q)]:
            assert not f.propose(bad,request,a.out/f'{target}{seed}',2)[0]
    raw=torch.zeros(1,4,6);raw[0,2,0]=10
    assert choose(raw,routes)[0]==2
    raw[0,2,0]=0
    assert choose(raw,routes)[0]==0
    save(a.out/'summary.json',dict(passed=True,candidate_bank_matches_v1=True,permutation_equivariant=True,
        malformed_and_stale_fallback=True,relative_repair_margin=True,trained_models_checked=4))
    print('RECOVERY_CONTRACT_PASSED',flush=True)


if __name__=='__main__':main()
