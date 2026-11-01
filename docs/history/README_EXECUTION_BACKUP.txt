P0 execution backup experiment, 2026-10-29

This is an isolated continuation of the frozen single-UAV perception experiment.
The EGO optimizer, EGO/ROG maps, route repair, envelope and stale/deadline gates
remain in place. The old run_perception.sh is the LEGACY protocol; use the new
commands below for the committed-backup executor. Never mix its results with the
old 280-case matrix.

Build in WSL Ubuntu-22.04:
  CMAKE_BUILD_PARALLEL_LEVEL=2 bash build_navigation.sh
  source /opt/ros/humble/setup.bash
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

Tests (each output directory must be new):
  python3 tools/test_perception_channel.py
  install/pointcloud_nav/lib/pointcloud_nav/route_manager_check
  python3 tools/test_backup_contract.py --out runs/backup-contract-new
  python3 tools/replay_backup_failures.py --baseline /path/to/original_failures --out runs/failure-replay-new
  python3 tools/run_backup_regression.py --baseline /path/to/original_failures --out runs/p0-new

The baseline argument is either the original perception-matrix-v1 or the
original_failures directory extracted from p0_evidence.tar.gz. It needs exactly
the two offset_wall_0_{ego,rog}_delay_400_r0 folders, including point clouds.
Binary cloud paths are relocated from their filenames for historical replay.

Single run:
  python3 tools/backup_trial.py --backend ego --condition baseline --out runs/demo-new
  python3 tools/backup_trial.py --backend ego --replay-prefix /path/to/original_failures/offset_wall_0_ego_delay_400_r0 --out runs/prefix-new

Worker extensions:
  plan with_backup=true returns both plan and backup_plan, or no_valid_backup.
  validate checks both curves and their P/V/A junction. prepare_backup=true is
  for historical-prefix initialization, under the same age/deadline gates.
  guard execution_schedule=[...] reports sampled invalid intervals over the
  complete scheduled execution (including its terminal hold). nominal_plan
  additionally diagnoses the unexecuted nominal tail. find_stop=true searches
  a finite family for a latest feasible stop witness, with a 40 ms loop budget.

Fixed P0 design parameters (frozen in worker.cpp and run manifest):
  nominal activation = trigger + 0.30 s; commit guard = 0.025 s.
  backup activation = nominal activation + 0.60 s (clipped to nominal end).
  uniform cubic spline, six controls; last three coincide at the stop point.
  knot intervals = .12/.16/.20/.25/.32/.40/.50/.65/.80/1.0/1.25 s.
  initial P/V/A exactly matched; velocity and acceleration control-point bounds
  and terminal rest checked by the original validator. No jerk limit is added.
  latest-stop witness grid = 50 ms, minimum activation lead = 0.30 s.
  a witness is not an optimal braking time; the 40 ms loop budget is not a hard
  real-time execution bound. Final admission still measures actual RPC time.

The executor commits nominal and backup together. New valid bundles replace
future commands; without renewal it follows the precommitted stop. A map update
rechecks the effective schedule, including pending activations. If no validated
escape remains it reports no_validated_escape and terminates as FAILURE, not
successful parking. Arbitrary newly appearing obstacles can invalidate a backup.

Map updates are serialized with guard/plan/validate requests. Guard can also run
while awaiting the next plan trigger; there is no independent safety thread.
The fixed 80 ms planning delay remains. Timeout tests inject 650 ms from t=3 s.
New-obstacle tests are abrupt synthetic geometry changes, not physical dynamic
object models. Ideal localization, instantaneous scans and ideal execution remain.

Historical-prefix branching replays maps through the original successful
validation request, computes a backup, accounts for its measured preparation
and publication time, checks the original activation deadline and latest frame,
then continues from that state. All observations after branching are generated
from the new path; original future scans never enter the continued flight.

Important limitations:
  5 ms centre/free + inflated occupancy sampling, no continuous swept-volume
  or full-body observed-free certificate; no voxel-age field or robust control.
  precommitted backup execution is not a new stale-map admission. Subsequent map
  updates can invalidate it. Unbounded computation cannot be made safe by this
  single serialized process; the execution model is ideal and not a flight stack.
  400 ms full-start runs can remain stationary due to stale_cloud; these are
  timeouts, not avoidance successes. Parking away from the goal is also timeout.
  No SUPER full safety system, learning policy, or paper novelty is claimed.
