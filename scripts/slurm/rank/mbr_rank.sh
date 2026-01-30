#!/bin/bash
#SBATCH --job-name=mbr-rank
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1,VramPerGpu:60GB                  # maximum run time.
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/mbr_rank_%j.out
#SBATCH --error=slurm_logs/mbr_rank_%j.err

set -euo pipefail
mkdir -p logs

# (Optional) modules for your cluster
module purge || true
module load GCCcore/11.3.0 || true
module load Python/3.10.4 || true
module load CUDA/12.1.1 || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Important: pin the venv so uv doesn't create ephemeral envs per job
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

if [[ $# -lt 1 ]]; then
  echo "Usage:"
  echo "  sbatch mbr_rank.slurm -- --model ... --prompts-file ... --outputs ... --out ... [--out-text ...] [other args]"
  exit 2
fi

echo "Command args: $*"
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Everything after -- is forwarded to python script.
uv run ./scripts/python/rank/mbr_rank.py "$@"

