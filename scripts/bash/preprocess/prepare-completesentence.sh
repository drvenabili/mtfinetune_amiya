# Directory for MT data
INPUT_DIR="./training_data/instruct_prompts/completesentence"
OUTPUT_DIR="./training_data/tokenized/complete"

tokenizer_path="${1}"
tokenizer=$(basename ${tokenizer_path})

mkdir -p "${OUTPUT_DIR}/${tokenizer}"
function prepare-instruct() {
  input_file="${1}"
  output_file="${2}"
  if [[ -d "${tokenizer_path}/config" ]]
  then
    tokenizer_path="${tokenizer_path}/config"
  fi
  sbatch ./scripts/slurm/preprocess/prepare-instruct.sh "${input_file}" "${tokenizer_path}" "${output_file}" 2048 " "
}

# only dialect to dialect mono
datasets=(egy.csv  jor.csv  mar.csv  pse.csv  sau.csv  syr.csv)
for data in "${datasets[@]}"
do
  mkdir -p "${OUTPUT_DIR}"/${tokenizer}/instruct-dialect-assistant-dialect
  input_file="${INPUT_DIR}"/instruct-dialect-assistant-dialect/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/instruct-dialect-assistant-dialect/${data::-4}
  if [[ -d ${output_file} ]]
  then
    echo "skipping ${output_file}"
    continue
  fi
  prepare-instruct ${input_file} ${output_file}
done

exit
##
datasets=(
  asr
  atlaset
  casablanca
  doda
  edc
  edgad
  goud
  joda
  madar26-train
  masccorpus
  saudial
  sauditweets
  shami
  ufal
)
# atlas  casablanca  doda  joda  madar26-train  saudial  ufal
training_dirs=(
#	instruct-dialect-assistant-dialect
	instruct-english-assistant-dialect
)


for data in "${datasets[@]}"
do
  for training_dir in "${training_dirs[@]}"
  do
    mkdir -p "${OUTPUT_DIR}"/${tokenizer}-complete/${data}
    for file in "${INPUT_DIR}"/${data}/${training_dir}/*.csv
    do
      file_name=$(basename ${file})
      input_file=${file}
      output_file="${OUTPUT_DIR}"/${tokenizer}-complete/${data}/${file_name::-4}
      if [[ -d ${output_file} ]]
      then
        echo "skipping ${output_file}"
        continue
      fi
      prepare-instruct ${input_file} ${output_file}
    done
  done
done

