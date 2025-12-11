#!/bin/bash

model_name=${1:-HuggingFaceTB/SmolLM3-3B}
org_name=${2:-unige-fti}

sbatch scripts/slurm/testupload/testupload.sh "$model_name" "$org_name"
