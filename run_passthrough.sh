#!/usr/bin/env bash
# run_passthrough.sh
# Slurm-compatible passthrough wrapper for Python training scripts
# Usage:
#   srun --gres=gpu:1 --cpus-per-task=4 \
#     bash run_passthrough.sh python src/nuscenes/resnet/minidrive_e2e.py --batch 32 --lr 1e-5

set -euo pipefail

# Load and activate conda environment
module load conda
conda activate catapult

echo "[run_passthrough] Active environment: $(conda info --envs | grep '*' )"
echo "[run_passthrough] Executing: $@"

# ensure tqdm doesn't break
export PYTHONUNBUFFERED=1

# Execute the passed command (e.g., python script)
exec "$@"
