#!/bin/sh
#SBATCH --job-name health           # this is a parameter to help you sort your job when listing it
#SBATCH --error health_logs_%j.error     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output health_logs_%j.out      # optional. By default the error and output files are merged
#SBATCH --mem 20GB
#SBATCH --time 00:05:00                  # maximum run time
#SBATCH --partition shared-gpu
#SBATCH --gres=gpu:1                 # maximum run time.


ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .env/bin/activate

srun uv run ./scripts/python/health.py
