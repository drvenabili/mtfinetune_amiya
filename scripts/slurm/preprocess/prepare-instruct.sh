#!/usr/bin/env bash
#SBATCH --job-name=tokenize_ds
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .venv/bin/activate

if [[ $# -lt 3 ]]; then
  echo "Usage:"
  echo "  sbatch preprocess-instruct.sh <csv_path> <model_id> <out_dir> [max_length] [system_prompt] [template_path] [override]"
  echo ""
  echo "override:"
  echo "  1 -> pass --override-existing-template"
  echo "  0 -> do not override (default)"
  echo ""
  echo "Example (no override):"
  echo "  sbatch tokenize-instruct.sh data/train.csv HuggingFaceTB/SmolLM3-3B tokenized_out 2048 \"You are a helpful assistant.\""
  echo ""
  echo "Example (template + override):"
  echo "  sbatch tokenize-instruct.sh data/train.csv HuggingFaceTB/SmolLM3-3B tokenized_out 2048 \"You are a helpful assistant.\" ./configs/minimal_template.jinja 1"
  exit 1
fi

CSV_PATH="${1}"
MODEL_ID="${2}"
OUT_DIR="${3}"
MAX_LENGTH="${4:-2048}"
SYSTEM_PROMPT="${5:-You are a helpful assistant.}"
TEMPLATE_PATH="${6:-}"          # optional
OVERRIDE_TEMPLATE="${7:-0}"     # optional (0/1)
SAVE_TOKENIZER_WITH_TEMPLATE="${8:-}"

TOKENIZE_SCRIPT="./scripts/python/preprocess/prepare-instruct.py"

# Make HF save in the cache folder
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Better CPU parallelism for tokenizers
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Optional CLI args
EXTRA_ARGS=()

if [[ -n "${TEMPLATE_PATH}" ]]; then
  EXTRA_ARGS+=(--template-path "${TEMPLATE_PATH}")
fi

if [[ "${OVERRIDE_TEMPLATE}" == "1" ]]; then
  EXTRA_ARGS+=(--override-existing-template)
fi

if [[ -n "${SAVE_TOKENIZER_WITH_TEMPLATE}" ]]
then
  EXTRA_ARGS+=(--save-tokenizer-with-template "${OUT_DIR}/tokenizer_with_template")
  
fi

echo "== Tokenizing =="
echo "CSV_PATH=${CSV_PATH}"
echo "MODEL_ID=${MODEL_ID}"
echo "OUT_DIR=${OUT_DIR}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "SYSTEM_PROMPT=${SYSTEM_PROMPT}"
echo "TEMPLATE_PATH=${TEMPLATE_PATH:-<none>}"
echo "OVERRIDE_TEMPLATE=${OVERRIDE_TEMPLATE}"
echo "CPUS=${SLURM_CPUS_PER_TASK}"

uv run "${TOKENIZE_SCRIPT}" "${CSV_PATH}" \
  --model-id "${MODEL_ID}" \
  --out-dir "${OUT_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --system-prompt "" \
  --delimiter "," \
  --encoding "utf-8" \
  --text-col-prompt "prompt" \
  --text-col-completion "completion" \
  --create-template-if-missing \
  --preview-n 10 \
  --save-config-json \
  --num-proc "${SLURM_CPUS_PER_TASK}" \
  "${EXTRA_ARGS[@]}"

echo "Done. Saved tokenized dataset to: ${OUT_DIR}"

