YOPO proposals + EGO optimizer/validator, 2026-10-29

This extends the existing single-uav-pointcloud-nav P0 source. It does NOT run
the upstream ROS1 simulator, controller or executable trajectory publisher.
Map backends, repair, rebound/refine, dynamic limits, terminal rest, scheduled
backup, freshness and latest-arrived-scan admission checks remain in place.

Upstream: https://github.com/TJU-Aerial-Robotics/YOPO
Branch: YOPO-Simple
Commit: bd8a7f91953cfef9f1aa9a48d7ff0ab7bceee132
Weights: YOPO/saved/YOPO_1/epoch50.pth, unchanged, loaded with weights_only=True.
Files and SHA256: third_party/yopo/upstream_manifest.json. MIT license retained.
No training or fine-tuning was performed. This is a hybrid adaptation, not a
reproduction of the complete YOPO paper or a claim of its original performance.

Runtime: Python 3.10; torch 2.4.1+cpu; numpy 1.21.5; scipy 1.8.0;
ruamel.yaml 0.17.21. CPU batch=1, torch/BLAS/OMP threads=1.
ROS2 Humble build dependencies unchanged. torchvision is not needed by the
vendored model. Initialization and five warm-up passes occur before flight.
All online preprocessing, inference, geometry decoding and diagnostic writes
are included in Worker.rpc_ms and therefore in the existing execution clock.

Model observation contract:
  Original model depth: 96x160, nominal 90x60 degree pinhole camera.
  Existing sensor: 360 degree, 17 layers (-8..8 degrees), 8 m range.
  A goal-oriented virtual view is sampled from the received organized ray cloud.
  Pixel rays use the nearest measured angular ray; axial depth is normalized
  by 20 m. No-return rays are capped at 8 m. Pixels outside vertical coverage
  are zero depth, never filled with free space. The observed mask is logged;
  the unchanged single-channel network itself does not receive an extra mask.
  Capture origin and capture P/V/A are retained. The network receives V/A and
  goal relative to that capture pose, using the author's normalization at 6 m/s.
  Future scans, scene geometry and ground truth are not given to the model.
  The network does not encode frame age; the existing 0.5 s external gate applies.

Adapter:
  Decode 15 author output primitives, rank by predicted score, take up to 3.
  Build the corresponding quintic geometric reference from capture P/V/A and
  predicted terminal P/V/A. Remove already-passed geometry, join the future
  handoff position, then let EGO independently assign spline timing and enforce
  handoff P/V/A and terminal rest. No polynomial coefficients are passed off as
  B-spline controls. Near the goal (<0.5 m), use repair directly.
  The C++ worker clips each proposal to the 3 m horizon / known-free prefix;
  requires >=0.3 m of prefix and >=0.1 m goal-distance improvement; then runs
  the existing rebound/refine, whole-curve check and backup feasibility check.
  If all model proposals fail, call the original repair route manager in the
  same request. Its time is not discarded. No threshold was relaxed.

Run (WSL Bash, project root):
  CMAKE_BUILD_PARALLEL_LEVEL=2 bash build_navigation.sh
  source /opt/ros/humble/setup.bash
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  PY=/home/chaonian/yopo-ego-20261029/venv/bin/python
  run_tag=$(date +%Y%m%d-%H%M%S)
  "$PY" tools/test_yopo_contract.py --out "runs/contracts-$run_tag"
  "$PY" tools/run_yopo_comparison.py --pilot --out "runs/pilot-$run_tag"
  "$PY" tools/run_yopo_comparison.py --repeats 2 --out "runs/comparison-$run_tag"

The full small comparison is 72 runs: 9 scene instances x 2 maps x 2 frontends
x 2 timing repetitions; 5 Hz x 12,240 points, zero transport delay, 80 ms fixed
planning delay, 35 s timeout. Scene seeds 211/212 were not in the original
perception matrix, but 211 is used by the pilot: do not call it a blind test.
Run directories cannot be overwritten. Failures remain in the summary.

Limitations: vertical-FOV and speed/horizon domain mismatch; no real yaw/control
dynamics; unchanged weights have no mask/age input; scores are for original
polynomial trajectories, not the post-EGO spline. Keeping the original mapping
and validator does not inherit YOPO's full-pipeline speed advantages. Map and
execution checks are sampled, not continuous swept-volume guarantees.
