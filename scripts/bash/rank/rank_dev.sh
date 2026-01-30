#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-}
BASE_OUTDIR=${2:-}
OUTDIR=${3:-}                 # e.g. ./output/SmolLM3-3B
METHOD=${4:-}              # path component in output tree
LANG_SET=${5:-}                # path component in output tree
DATASET_DIRNAME=${6:-}        # path component in output tree (madar)
TEST_DIR=${7:-./test_data}         # prompts live here

if [[ -z "${MODEL}" || -z "${BASE_OUTDIR}" || -z "${OUTDIR}" ]]; then
  echo "Usage: $0 <MODEL> <BASE_OUTDIR> <OUTDIR> [METHOD=fidelity] [LANG_SET=mono] [DATASET_DIRNAME=madar] [TEST_DIR=./test_data]"
  echo "Example:"
  echo "  $0 HuggingFaceTB/SmolLM3-3B ./output/SmolLM3-3B fidelity mono madar ./test_data"
  exit 1
fi

# dialects derived from files in test_data (egy.csv -> egy)
mapfile -t DIALECTS < <(ls -1 "${TEST_DIR}"/*.csv 2>/dev/null | xargs -n1 basename | sed 's/\.csv$//' | sort)

if [[ ${#DIALECTS[@]} -eq 0 ]]; then
  echo "No .csv found in ${TEST_DIR}"
  exit 1
fi

echo "MODEL=${MODEL}"
echo "BASE_OUTDIR=${BASE_OUTDIR}"
echo "TEST_DIR=${TEST_DIR}"
echo "DIALECTS=${DIALECTS[*]}"
echo ""

# We expect run dirs like:
#   ${BASE_OUTDIR}_<topP>_<temp>/<METHOD>/<LANG_SET>/<DATASET_DIRNAME>/<dialect>.out
#OUT_GLOB_PREFIX="${BASE_OUTDIR}"_*_*/"${METHOD}"/"${LANG_SET}"/"${DATASET_DIRNAME}"

#OUT_GLOB_PREFIX=./output/{SmolLM3-3B,mt-fidelity-all-small-template-lr_3e-5-mt}_*_*/"${METHOD}"/"${LANG_SET}"/"${DATASET_DIRNAME}"
OUT_GLOB_PREFIX="${BASE_OUTDIR}"_*_*/"${METHOD}"/"${LANG_SET}"/"${DATASET_DIRNAME}"

for d in "${DIALECTS[@]}"; do
  PROMPTS_FILE="${TEST_DIR}/${d}.csv"
  [[ -f "$PROMPTS_FILE" ]] || { echo "Missing ${PROMPTS_FILE}"; continue; }

  # collect candidate outputs for this dialect across hyperparam runs
  mapfile -t OUT_FILES < <(eval "ls -1 ${OUT_GLOB_PREFIX}/${d}.out" 2>/dev/null || true)

  if [[ ${#OUT_FILES[@]} -eq 0 ]]; then
    echo "No outputs found for ${d}: ${OUT_GLOB_PREFIX}/${d}.out"
    continue
  fi

  OUT_CSV="${OUTDIR}/${d}.results.csv"
  OUT_TXT="${OUTDIR}/${d}.best.txt"

  echo "→ ${d}: ${#OUT_FILES[@]} candidates"
  echo "   prompts: ${PROMPTS_FILE}"
  echo "   out_csv: ${OUT_CSV}"
  echo "   out_txt: ${OUT_TXT}"

  # build --outputs args safely
  OUTPUT_ARGS=()
  for f in "${OUT_FILES[@]}"; do
    OUTPUT_ARGS+=(--outputs "$f")
  done
  if [[ -f "${OUT_CSV}" && -f "${OUT_TXT}" ]]
  then
    echo "skip ${d}"
    continue
  fi
  sbatch \
    --job-name="mbrT_${d}" \
    --output="slurm_logs/mbr_${d}_%j.out" \
    --error="slurm_logs/mbr_${d}_%j.err" \
    ./scripts/slurm/rank/mbr_rank.sh --model "${MODEL}" \
      --prompts-file "${PROMPTS_FILE}" \
      "${OUTPUT_ARGS[@]}" \
      --length-norm \
      --out "${OUT_CSV}" \
      --out-text "${OUT_TXT}"
done

echo ""
echo "Submitted ranking jobs. Results will appear in: ${OUTDIR}/"

