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

### the arguments of the bash
method=${1}
model=${2}
dataset=${3}

#### print arguments ####
echo "method=${method}"
echo "model-path=${model}"
echo "dataset=${dataset}"

srun uv run scripts/python/finetune/finetune.py --method "$method" --model-name "$model" --dataset-name "$dataset"