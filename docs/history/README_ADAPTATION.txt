2026-10-30: YOPO limitations and lightweight range-scoring experiment

This is an isolated extension of the existing YOPO + EGO + P0 executor.
EGO optimization, sampled spline validation, maps, body/configuration, and
execution/backup/commit semantics are unchanged. The original frozen project is
/home/chaonian/yopo-ego-20261029/single-uav-pointcloud-nav.

New frontends: tools/adapted_frontends.py
  aligned: original YOPO weights, future-start endpoint-direction reference,
           3 m EGO horizon and 3 m near-goal repair handoff.
  rule: 15 fixed references ranked using delivered-ray geometry.
  learn11/22/33: 47,976-parameter CNN/MLP scorer trained from scratch; same
                15 references plus repair action. These are NOT fine-tuned YOPO
                weights or a reproduction of original guidance learning.

Observation: three channels (range/hit/valid), 17 x 72 spherical range image,
22 capture/handoff/goal/age/budget state features. Uses only already-delivered
clouds. Always pass accepted proposals through the original EGO and validators.
The current experiment does not establish continuous swept-volume safety.

Data: 399 same-state examples, each with 16 branch evaluations; 320 training,
79 validation. Forest seeds 3101-3106 train, 3201/3202 validation, 3301-3304
test. Old offset_wall/forest211 are separately identified regressions.
Training seeds: 11,22,33. CPU single-thread,80 epochs, validation selection.
All failed episodes and negative results are retained.

WSL example:
source /opt/ros/humble/setup.bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/chaonian/yopo-ego-20261029/venv/bin/python
CMAKE_BUILD_PARALLEL_LEVEL=2 bash build_navigation.sh
run_tag=$(date +%Y%m%d-%H%M%S)
"$PY" tools/test_adaptation_contract.py --models --out "runs/contracts-$run_tag"
"$PY" tools/run_adaptation.py --stage pilot --out "runs/pilot-$run_tag"
"$PY" tools/run_adaptation.py --stage test --out "runs/test-$run_tag"

Training (not needed to reuse included models):
"$PY" tools/run_adaptation.py --stage collect --out "runs/collect-$run_tag"
"$PY" tools/train_range_scorer.py collect --run "runs/collect-$run_tag" --out "runs/labels-$run_tag"
"$PY" tools/train_range_scorer.py train --dataset "runs/labels-$run_tag" --out "models-retrain-$run_tag"
The new output cannot overwrite frozen models. Connect retrained weights in
another isolated copy and freeze again before evaluation.

Full Chinese results, limitations, sources, commands, and archive hashes reside
in the parent Windows delivery. No improvement over repair is claimed.
