#!/bin/bash

method=${1:-trl}
model_name=${2:-HuggingFaceTB/SmolLM3-3B}
dataset_name=${3:-fillwith/realdata}

sbatch ./scripts/slurm/finetune/finetune.sh "$method" "$model_name" "$dataset_name"