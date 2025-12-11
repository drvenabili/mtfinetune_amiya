#!/bin/sh
#SBATCH --job-name testupload
#SBATCH --error testupload_logs_%j.error
#SBATCH --output testupload_logs_%j.out
#SBATCH --mem 20GB
#SBATCH --time 01:00:00
#SBATCH --partition shared-gpu
#SBATCH --gres=gpu:1

ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .env/bin/activate

model_name=${1:-HuggingFaceTB/SmolLM3-3B}
org_name=${2:-unige-fti}

srun uv run scripts/python/testupload/testupload.py --model-name "$model_name" --org-name "$org_name"
