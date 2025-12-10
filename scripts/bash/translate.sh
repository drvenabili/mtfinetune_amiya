model=${1}
output_directory=${2}

if [[ -z ${model} ]]
then
  echo "variable model not set"
  exit
fi

if [[ -z ${output_directory} ]]
then
  echo "please, put the output directory"
  exit
fi

if [ ! -d ${output_directory} ]
then
  mkdir -p ${output_directory}/btc
  mkdir -p ${output_directory}/flores
fi

echo "model=${1}"
echo "output=${2}"
# MADAR dataset
# only interested in:
  # ary - Morocco
  # egy - Egypt
  # 
madar=(dza-eng.csv egy-eng.csv eng-dza.csv eng-mar.csv eng-sau.csv eng-syr.csv mar-msa.csv msa-egy.csv msa-pse.csv msa-sdn.csv pse-eng.csv sau-eng.csv sdn-eng.csv syr-eng.csv dza-msa.csv egy-msa.csv eng-egy.csv eng-pse.csv eng-sdn.csv mar-eng.csv msa-dza.csv msa-mar.csv msa-sau.csv msa-syr.csv pse-msa.csv sau-msa.csv sdn-msa.csv syr-msa.csv)

for data in "${madar[@]}"
do
  # we do not need to translate if it's already translated
  if [[ -f ${data::-4}.out ]]
  then
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ./data/bi/btec/madar26/${data} ${output_directory}/btc/${data::-4}.out ${model}
done

### FLORES dataset
flores=(egy-eng.csv eng-sau.csv msa-mar.csv pse-msa.csv egy-msa.csv eng-syr.csv msa-pse.csv sau-eng.csv eng-egy.csv mar-eng.csv msa-sau.csv sau-msa.csv eng-mar.csv mar-msa.csv msa-syr.csv syr-eng.csv eng-pse.csv msa-egy.csv pse-eng.csv syr-msa.csv)

for data in "${flores[@]}"
do
  # we do not need to translate if it's already translated
  if [[ -f ${data::-4}.out ]]
  then
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ./data/bi/wiki/flores-dev/${data} ${output_directory}/flores/${data::-4}.out ${model}
done

