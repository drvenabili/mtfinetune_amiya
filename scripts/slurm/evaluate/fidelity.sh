#!/usr/bin/env bash
#SBATCH --job-name=score_eval
#SBATCH --mem=20GB
#SBATCH --time=01:30:00
#SBATCH --partition shared-gpu
#SBATCH --constraint=COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE|COMPUTE_TYPE_ADA
#SBATCH --gres=gpu:1                  # maximum run time.

ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .venv/bin/activate
export TRANSFORMERS_VERBOSITY=error

##### Args ####
if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch fidelity.sh <output_model_dir>"
  exit 1
fi
output_model=${1}

#### parameters output
#echo "generations_directory=${output_model}"


function score_adi() {
 if [ $# -ne 2 ]; then
    echo "Error: Exactly 2 arguments required!"
    return 1
  fi
  output=$(uv run ./scripts/python/evaluate/evaluator.py score-adi "$1" --dialect "$2" 2>/dev/null || true)
  NADI=$(echo "$output" | jq -r '.prob // 0.0' 2>/dev/null)
  ALDI=$(echo "$output" | jq -r '.dialectness // 0.0' 2>/dev/null)
  ADI=$(echo "$output" | jq -r '.score // 0.0' 2>/dev/null)
  MACRO_ADI=$(echo "$output" | jq -r '.macro_score // 0.0' 2>/dev/null)
  echo -e "${NADI:-0.0} ${ALDI:-0.0} ${ADI:-0.0} ${MACRO_ADI:-0.0}"
}

###################################
#### Fidelity Evaluation ##########
###################################
output_fidelity=${output_model}/fidelity

### output monolingual
output_fidelity_monolingual=${output_fidelity}/mono

###### MADAR
output_files=${output_fidelity_monolingual}/madar
prompt_files=./data/mono/btec/madar26/
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)
for data in "${data_files[@]}"
do
  dialect=${data::-4}
  reference_file=${prompt_files}/${data}
  score_adi ${output_files}/${dialect}.out ${dialect}
done

##### habibi
output_files=${output_fidelity_monolingual}/habibi
prompt_files=./data/mono/music/habibi
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  dialect=${data::-4}
  reference_file=${prompt_files}/${data}
  score_adi ${output_files}/${dialect}.out ${dialect}
done

##### flores
output_files=${output_fidelity_monolingual}/flores-dev

prompt_files=./data/mono/wiki/flores-dev
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  dialect=${data::-4}
  reference_file=${prompt_files}/${data}
  score_adi ${output_files}/${dialect}.out ${dialect}
done

###################################
##### Crosslingual ################
###################################
output_fidelity_cross=${output_fidelity}/cross

## hehe
output_files=${output_fidelity_cross}/hehe

prompt_files=./data/xling/hehe
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  dialect=${data::-4}
  output_file=${output_files}/${dialect}.out
  # we do not need to translate if it's already translated
  score_adi ${output_file} ${dialect}
done

## okapi
output_files=${output_fidelity_cross}/okapi

prompt_files=./data/xling/okapi
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  dialect=${data::-4}
  output_file=${output_files}/${dialect}.out
  # we do not need to translate if it's already translated
  score_adi ${output_file} ${dialect}
done

## sharegpt
output_files=${output_fidelity_cross}/sharegpt
prompt_files=./data/xling/sharegpt
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  dialect=${data::-4}
  output_file=${output_files}/${dialect}.out
  # we do not need to translate if it's already translated
  score_adi ${output_file} ${dialect}
done

