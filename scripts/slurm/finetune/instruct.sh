#!/usr/bin/env bash
#SBATCH --job-name=sft
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:4,VramPerGpu:70GB                  # maximum run time.
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --requeue

set -euo pipefail
mkdir -p slurm_logs

CONFIG_YAML=${1:?Usage: sbatch run_sft.slurm config.yaml [--override key=value ...]}
shift || true
OVERRIDES=("$@")   # everything after config.yaml is forwarded (e.g., -o ...)

module purge || true
module load GCCcore/11.3.0 || true
module load Python/3.10.4 || true
module load CUDA/12.1.1 || true

# ---- Paths / fallbacks (avoid set -u crashes) ----
# Prefer SCRATCH if provided by the cluster, otherwise fall back to TMPDIR or $HOME/scratch
SCRATCH_DIR="${SCRATCH:-${TMPDIR:-$HOME/scratch}}"
mkdir -p "$SCRATCH_DIR"

export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Important: pin the venv so uv doesn't create ephemeral envs per job
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

# Good: keep caches on scratch
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH_DIR/uv_cache}"

# Optional: avoid any “helpful” auto-upgrades mid-run
export UV_FROZEN=1

echo "JobID=${SLURM_JOB_ID} Node=${SLURMD_NODENAME}"
nvidia-smi || true

uv run python -u ./scripts/python/finetune/instruct.py "${CONFIG_YAML}" "${OVERRIDES[@]}"

