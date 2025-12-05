#!/bin/sh
#SBATCH --job-name decode           # this is a parameter to help you sort your job when listing it
#SBATCH --error decode_logs_%j.error     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output decode_logs_%j.out      # optional. By default the error and output files are merged
#SBATCH --mem 20GB
#SBATCH --time 01:00:00                  # maximum run time
#SBATCH --partition shared-gpu
#SBATCH --gres=gpu:1,VramPerGpu:24GB                  # maximum run time.

### loading the modules
ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .env/bin/activate

### the arguments of the bash
prompt=${1}
model=${2}
method=${3}
max_new_tokens=${4:-256}
temperature=${5:-0.7}
top_p=${6:-0.9}
seed=${7:-111}

#### print arguments ####
echo "prompt=${prompt}"
echo "model-path=${model}"
echo "method=${method}"
echo "max-new-tokens=${max_new_tokens}"
echo "temperature=${temperature}"
echo "top-p=${top_p}"
echo "seed=${seed}"
########################

srun uv run ./scripts/python/generate/generate.py generate \
  "${prompt}" \
  --model-path ${model} \
  --method ${method} \
  --max-new-tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --top-p ${top_p} \
  --seed ${seed}

