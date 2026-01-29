model=${1}
output_directory=${2}
method=${3}
max_new_tokens=${4:-512}
temperature=${5:-0.7}
top_p=${6:-0.9}

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

## variables for the model
echo "model=${1}"
echo "output=${2}"

# MADAR dataset
# only interested in:
  # ary - Morocco
  # egy - Egypt
  #

###################################
#### Machine translation ##########
###################################

# output for MT
output_mt=${output_directory}/MT
mkdir -p ${output_mt}

# output for madar corpus
output_mt_madar=${output_mt}/madar
mkdir -p ${output_mt_madar}

madar=(egy-eng.csv eng-mar.csv eng-sau.csv eng-syr.csv mar-msa.csv msa-egy.csv msa-pse.csv pse-eng.csv sau-eng.csv syr-eng.csv egy-msa.csv eng-egy.csv eng-pse.csv mar-eng.csv msa-mar.csv msa-sau.csv msa-syr.csv pse-msa.csv sau-msa.csv syr-msa.csv)

for data in "${madar[@]}"
do
  output_file=${output_mt_madar}/${data::-4}.out
  # we do not need to translate if it's already translated
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ./data/bi/btec/madar26/${data} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
done

### FLORES dataset
# output for flores corpus
output_mt_flores=${output_mt}/flores
mkdir -p ${output_mt_flores}

flores=(egy-eng.csv eng-sau.csv msa-mar.csv pse-msa.csv egy-msa.csv eng-syr.csv msa-pse.csv sau-eng.csv eng-egy.csv mar-eng.csv msa-sau.csv sau-msa.csv eng-mar.csv mar-msa.csv msa-syr.csv syr-eng.csv eng-pse.csv msa-egy.csv pse-eng.csv syr-msa.csv)

for data in "${flores[@]}"
do
  output_file=${output_mt_flores}/${data::-4}.out
  # we do not need to translate if it's already translated
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ./data/bi/wiki/flores-dev/${data} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
done
 
 
###################################
#### Fidelity Generation ##########
###################################
# output for fidelity
output_fidelity=${output_directory}/fidelity
mkdir -p ${output_fidelity}

###################################
##### Monolingual #################
###################################
output_fidelity_monolingual=${output_fidelity}/mono
## MADAR
output_files=${output_fidelity_monolingual}/madar
mkdir -p ${output_files}

prompt_files=./data/mono/btec/madar26
data_files=(egy.csv  mar.csv  pse.csv  sau.csv syr.csv)
for data in "${data_files[@]}"
do
  output_file=${output_files}/${data::-4}.out
  # we do not need to translate if it's already translated
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ${prompt_files}/${data} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
done

## habibi
output_files=${output_fidelity_monolingual}/habibi
mkdir -p ${output_files}

prompt_files=./data/mono/music/habibi
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)
for data in "${data_files[@]}"
do
  output_file=${output_files}/${data::-4}.out
  # we do not need to translate if it's already translated
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ${prompt_files}/${data} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
done

## flores
output_files=${output_fidelity_monolingual}/flores-dev
mkdir -p ${output_files}

prompt_files=./data/mono/wiki/flores-dev
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  output_file=${output_files}/${data::-4}.out
  # we do not need to translate if it's already translated
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ${prompt_files}/${data} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
done

###################################
##### Crosslingual ################
###################################
output_fidelity_cross=${output_fidelity}/cross


## hehe
output_files=${output_fidelity_cross}/hehe
mkdir -p ${output_files}

prompt_files=./data/xling/hehe
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  output_file=${output_files}/${data::-4}.out
  # we do not need to translate if it's already translated
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ${prompt_files}/${data} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
done

## okapi
output_files=${output_fidelity_cross}/okapi
mkdir -p ${output_files}

prompt_files=./data/xling/okapi
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  output_file=${output_files}/${data::-4}.out
  # we do not need to translate if it's already translated
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ${prompt_files}/${data} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
done

## sharegpt
output_files=${output_fidelity_cross}/sharegpt
mkdir -p ${output_files}

prompt_files=./data/xling/sharegpt
data_files=(egy.csv mar.csv pse.csv sau.csv syr.csv)

for data in "${data_files[@]}"
do
  output_file=${output_files}/${data::-4}.out
  # we do not need to translate if it's already translated
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  sbatch ./scripts/slurm/generate/generate-batch.sh ${prompt_files}/${data} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
done

