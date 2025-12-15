#!/usr/bin/env bash
#SBATCH --job-name=score_eval
#SBATCH --error=decode_logs_%j.error
#SBATCH --output=decode_logs_%j.out
#SBATCH --mem=20GB
#SBATCH --time=00:20:00
#SBATCH --partition=shared-cpu

export TRANSFORMERS_VERBOSITY=error

##### Args ####
if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch fidelity.sh <output_model_dir>"
  exit 1
fi
output_model=${1}

#### parameters output
echo "generations_directory=${output_model}"


function score_adi() {
 if [ $# -ne 2 ]; then
    echo "Error: Exactly 2 arguments required!"
    return 1
  fi
  output=$(uv run ./scripts/python/evaluate/evaluator.py score-adi ${1} --dialect ${2})
  NADI=$(echo ${output} | jq -r '.prob')
  ALDI=$(echo ${output} | jq -r '.dialectness')
  ADI=$(echo ${output} | jq -r '.score')
  MACRO_ADI=$(echo ${output} | jq -r '.macro_score')
  echo -e ${NADI}" "${ALDI}" "${ADI}" "${MACRO_ADI}
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
data_files=(dza.csv  egy.csv  mar.csv  pse.csv  sau.csv  sdn.csv  syr.csv)

for data in "${data_files[@]}"
do
  dialect=${data::-4}
  reference_file=${prompt_files}/${data}
  score_adi ${output_files}/${dialect}.out ${dialect}
done

##### habibi
output_files=${output_fidelity_monolingual}/habibi

prompt_files=./data/mono/music/habibi
data_files=(dza.csv egy.csv kwt.csv mar.csv pse.csv sau.csv sdn.csv syr.csv)

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
data_files=(dza.csv egy.csv kwt.csv mar.csv pse.csv sau.csv sdn.csv syr.csv)

for data in "${data_files[@]}"
do
  dialect=${data::-4}
  output_file=${output_files}/${dialect}.out
  # we do not need to translate if it's already translated
  score_adi ${output_file} ${dialect}
done

echo "okapi"
## okapi
output_files=${output_fidelity_cross}/okapi

prompt_files=./data/xling/okapi
data_files=(dza.csv egy.csv kwt.csv mar.csv pse.csv sau.csv sdn.csv syr.csv)

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
data_files=(dza.csv egy.csv kwt.csv mar.csv pse.csv sau.csv sdn.csv syr.csv)

for data in "${data_files[@]}"
do
  dialect=${data::-4}
  output_file=${output_files}/${dialect}.out
  # we do not need to translate if it's already translated
  score_adi ${output_file} ${dialect}
done

