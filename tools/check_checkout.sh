#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
py=${PYTHON:-python3}
out=${1:-runs/checkout-$(date +%Y%m%d-%H%M%S)}
test ! -e "$out"
mkdir -p "$out"
"$py" tools/test_perception_channel.py
install/pointcloud_nav/lib/pointcloud_nav/route_manager_check
"$py" tools/test_d435i_contract.py --out "$out/camera"
"$py" tools/test_backup_contract.py --out "$out/backup"
"$py" tools/train_recovery.py --dataset datasets/recovery-v4 --out "$out/train-full" --kinds mlp --seeds 11 --epochs 1
"$py" tools/train_recovery.py --dataset datasets/recovery-v4 --out "$out/train-base" --kinds spatial --seeds 11 --epochs 1 --base-only
