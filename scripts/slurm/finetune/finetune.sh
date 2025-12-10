#!/bin/sh
#SBATCH --job-name finetune
#SBATCH --error finetune_logs_%j.error
#SBATCH --output finetune_logs_%j.out
#SBATCH --mem 20GB
#SBATCH --time 02:00:00
#SBATCH --partition shared-gpu
#SBATCH --gres=gpu:1

ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .env/bin/activate

method=${1:-trl}
model_name=${2:-HuggingFaceTB/SmolLM3-3B}

srun uv run scripts/python/finetune/finetune.py --method "$method" --model-name "$model_name"
