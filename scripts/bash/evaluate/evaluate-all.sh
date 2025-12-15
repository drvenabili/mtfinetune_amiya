#!/usr/bin/env bash
##### Args ####
if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch score_eval.slurm <output_model_dir>"
  exit 1
fi

output_dir=${1}
logs_name="$(basename "${output_dir}")"

sbatch \
  --output=${logs_name}_fidelity.out \
  --error=${logs_name}_fidelity.err \
  ./scripts/slurm/evaluate/fidelity.sh ${output_dir}

sbatch \
  --output=${logs_name}_mt.out \
  --error=${logs_name}_mt.err \
   ./scripts/slurm/evaluate/translation.sh ${output_dir}

