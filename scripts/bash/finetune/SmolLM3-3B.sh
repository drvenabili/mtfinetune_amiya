#!/bin/bash

method=${1:-trl}
model_name=${2:-HuggingFaceTB/SmolLM3-3B}

sbatch ./scripts/slurm/finetune/finetune.sh "$method" "$model_name"