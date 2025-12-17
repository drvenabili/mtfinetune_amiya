#!/usr/bin/env bash
#SBATCH --job-name=score_eval
#SBATCH --mem=20GB
#SBATCH --time=00:05:00
#SBATCH --partition=shared-cpu
ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
source .venv/bin/activate

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch translation.sh <output_model_dir>"
  exit 1
fi
output_model=${1}

#### parameters output
#echo "generations_directory=${output_model}"

##### Light version of the scores (less time)
function score_mt() {
 if [ $# -ne 2 ]; then
    echo "Error: Exactly 2 arguments required!"
    return 1
  fi
  output=$(sacrebleu ${2} -i ${1} -m chrf --chrf-char-order 6 --chrf-word-order 2 -w6 -b)
# ChrF_corpus_score
  echo ${output}
}

###################################
######## MT Evaluation ############
###################################
# output for MT
output_mt=${output_model}/MT

#### MADAR 
output_files=${output_mt}/madar
madar=(egy-eng.csv eng-mar.csv eng-sau.csv eng-syr.csv mar-msa.csv msa-egy.csv msa-pse.csv pse-eng.csv sau-eng.csv syr-eng.csv egy-msa.csv eng-egy.csv eng-pse.csv mar-eng.csv msa-mar.csv msa-sau.csv msa-syr.csv pse-msa.csv sau-msa.csv syr-msa.csv)

for data in "${madar[@]}"
do
  pair_language=${data::-4}
  reference_file=./data/bi/btec/madar26/references/${pair_language}
  score_mt ${output_files}/${data::-4}.out ${reference_file} 
done

#### FLORES

output_mt_flores=${output_mt}/flores

flores=(egy-eng.csv eng-sau.csv msa-mar.csv pse-msa.csv egy-msa.csv eng-syr.csv msa-pse.csv sau-eng.csv eng-egy.csv mar-eng.csv msa-sau.csv sau-msa.csv eng-mar.csv mar-msa.csv msa-syr.csv syr-eng.csv eng-pse.csv msa-egy.csv pse-eng.csv syr-msa.csv)

for data in "${flores[@]}"
do
  pair_language=${data::-4}
  reference_file=./data/bi/wiki/flores-dev/references/${pair_language}
  score_mt ${output_mt_flores}/${data::-4}.out ${reference_file} 
done

