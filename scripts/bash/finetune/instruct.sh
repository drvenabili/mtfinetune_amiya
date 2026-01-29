#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash instuct.sh 10 ./scripts/slurm/finetune/instruct.sh config.yaml afterany
#   bash instruct.sh 10 ./scripts/slurm/finetune/instruct.sh config.yaml afterany \
#       -o training.learning_rate=2e-5 -o output_dir=runs/lr_2e-5
#
# Anything after the 4th argument is forwarded to the SLURM script (overrides, etc.)

N=${1:-5}                                        # number of chained jobs
SLURM_SCRIPT=${2:-./scripts/slurm/finetune/instruct.sh}
CONFIG=${3:-config.yaml}
DEPEND_MODE=${4:-afterok}                         # or afterany
shift $(( $# >= 4 ? 4 : $# )) || true             # drop first 4 args if present

EXTRA_ARGS=("$@")                                 # forwarded (e.g., -o key=value ...)

jid=""
for i in $(seq 1 "$N"); do
  if [[ -z "$jid" ]]; then
    jid=$(sbatch --parsable "${SLURM_SCRIPT}" "${CONFIG}" "${EXTRA_ARGS[@]}")
  else
    jid=$(sbatch --parsable --dependency="${DEPEND_MODE}:${jid}" "${SLURM_SCRIPT}" "${CONFIG}" "${EXTRA_ARGS[@]}")
  fi
  echo "Queued job $i: $jid"
done

