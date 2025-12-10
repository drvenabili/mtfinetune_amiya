#!/bin/sh
#SBATCH --job-name decode           # this is a parameter to help you sort your job when listing it
#SBATCH --error decode_logs_%j.error     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output decode_logs_%j.out      # optional. By default the error and output files are merged
#SBATCH --mem 20GB
#SBATCH --time 01:00:00                  # maximum run time
#SBATCH --partition shared-gpu
#SBATCH --gres=gpu:1,VramPerGpu:24GB                  # maximum run time.

### loading the modules
export CUDA_LAUNCH_BLOCKING=1
ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .env/bin/activate

### the arguments of the bash
prompts_file=${1}
output_file=${2}
model=${3}
method=${4:-base}
max_new_tokens=${5:-256}
temperature=${6:-0.7}
top_p=${7:-0.9}
seed=${8:-111}

#### print arguments ####
echo "prompts_file=${prompts_file}"
echo "model-path=${model}"
echo "method=${method}"
echo "max-new-tokens=${max_new_tokens}"
echo "temperature=${temperature}"
echo "top-p=${top_p}"
echo "seed=${seed}"
if [ -z ${output_file} ]
then
  echo "output_file=${output_file}"
fi
########################

srun uv run ./scripts/python/generate/generate.py generate-batch \
  "${prompts_file}" \
  --model-path ${model} \
  --method ${method} \
  --max-new-tokens ${max_new_tokens} \
  --temperature ${temperature} \
  --top-p ${top_p} \
  --seed ${seed} \
  --output-file ${output_file}

