#!/usr/bin/env bash
#SBATCH --job-name=score_eval
#SBATCH --mem=20GB
#SBATCH --time=00:20:00
#SBATCH --partition=shared-cpu
ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .venv/bin/activate

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch translation.sh <output_model_dir>"
  exit 1
fi
output_model=${1}

#### parameters output
echo "generations_directory=${output_model}"

function score_mt() {
 if [ $# -ne 2 ]; then
    echo "Error: Exactly 2 arguments required!"
    return 1
  fi
  output=$(uv run ./scripts/python/evaluate/evaluator.py score-mt ${1} ${2})
# ChrF_corpus_score
  chrf=$(echo ${output} | jq -r '.ChrF_corpus_score')
  bleu=$(echo ${output} | jq -r '.SpBLEU_corpus_score')
  echo -e ${chrf}" "${bleu}
}

###################################
######## MT Evaluation ############
###################################
# output for MT
output_mt=${output_model}/MT

#### MADAR 
output_files=${output_mt}/madar
madar=(dza-eng.csv egy-eng.csv eng-dza.csv eng-mar.csv eng-sau.csv eng-syr.csv mar-msa.csv msa-egy.csv msa-pse.csv msa-sdn.csv pse-eng.csv sau-eng.csv sdn-eng.csv syr-eng.csv dza-msa.csv egy-msa.csv eng-egy.csv eng-pse.csv eng-sdn.csv mar-eng.csv msa-dza.csv msa-mar.csv msa-sau.csv msa-syr.csv pse-msa.csv sau-msa.csv sdn-msa.csv syr-msa.csv)

for data in "${madar[@]}"
do
  reference_file=./data/bi/btec/madar26/${data}
  score_mt ${output_files}/${data::-4}.out ${reference_file} 
done

#### FLORES

output_mt_flores=${output_mt}/flores

flores=(egy-eng.csv eng-sau.csv msa-mar.csv pse-msa.csv egy-msa.csv eng-syr.csv msa-pse.csv sau-eng.csv eng-egy.csv mar-eng.csv msa-sau.csv sau-msa.csv eng-mar.csv mar-msa.csv msa-syr.csv syr-eng.csv eng-pse.csv msa-egy.csv pse-eng.csv syr-msa.csv)

for data in "${flores[@]}"
do
  reference_file=./data/bi/wiki/flores-dev/${data}
  score_mt ${output_mt_flores}/${data::-4}.out ${reference_file} 
done

