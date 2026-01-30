model=${1}
output_directory=${2}
method=${3}
max_new_tokens=${4:-512}
temperature=${5:-0.7}
top_p=${6:-0.9}
seed=${7:-42}
GPU=${8:-50GB}

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

function generate-batch() {
 if [ $# -ne 2 ]; then
    echo "Error: Exactly 2 arguments required!"
    return 1
  fi
  if [[ -f ${output_file} ]]
  then
    echo "skipping ${output_file}"
    return
  fi
  input_file=${1}
  output_file=${2}
  sbatch --gres=gpu:1,VramPerGpu:${GPU} ./scripts/slurm/generate/generate-batch.sh ${input_file} ${output_file} ${model} ${method} ${max_new_tokens} ${temperature} ${top_p} ${seed}
}



## variables for the model
echo "model=${1}"
echo "output=${2}"

output_test=${output_directory}/test
mkdir -p ${output_test}

INPUT_DIR="./test_data"

datasets=(egy.csv  mor.csv  pes.csv  sau.csv  syr.csv)
for data in "${datasets[@]}"
do
  output_file="${output_test}"/${data::-4}.out
  input_file="${INPUT_DIR}"/${data}
  generate-batch ${input_file} ${output_file}
done
