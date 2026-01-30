

# Directory for MT data
INPUT_DIR="./training_data/instruct_prompts/bi"
OUTPUT_DIR="./training_data/tokenized/bi"

tokenizer_path="${1}"
tokenizer=$(basename ${tokenizer_path})

mkdir -p "${OUTPUT_DIR}/${tokenizer}"
function prepare-instruct() {
  input_file="${1}"
  output_file="${2}"
  if [[ -d ${output_file} ]]
  then
    echo "skipping ${output_file}"
    return
  fi
  if [[ -d "${tokenizer_path}/config" ]]
  then
    tokenizer_path="${tokenizer_path}/config"
    echo "using ${tokenizer_path}"
  fi 
  sbatch ./scripts/slurm/preprocess/prepare-instruct.sh "${input_file}" ${tokenizer_path} "${output_file}" 2048 " "
}

##
# atlas  casablanca  doda  joda  madar26-train  saudial  ufal

training_dir="atlas"
training_files=(eng-mar-new.csv mar-eng-new.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-8}
  echo "${output_file}"
  prepare-instruct ${input_file} ${output_file}
done



training_dir="casablanca"
training_files=(eng-pse-new.csv  pse-eng-new.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-8}
  prepare-instruct ${input_file} ${output_file}
done


training_dir="doda"
training_files=(eng-mar-new.csv  mar-eng-new.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-8}
  prepare-instruct ${input_file} ${output_file}
done


training_dir="joda"
training_files=(jor-msa-new.csv  msa-jor-new.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-8}
  prepare-instruct ${input_file} ${output_file}
done


training_dir="madar26-train"
training_files=(dza-eng.csv  egy-eng.csv  eng-dza.csv  eng-mar.csv  eng-sau.csv  eng-syr.csv  mar-msa.csv  msa-egy.csv  msa-pse.csv  msa-sdn.csv  pse-eng.csv  sau-eng.csv  sdn-eng.csv  syr-eng.csv
dza-msa.csv  egy-msa.csv  eng-egy.csv  eng-pse.csv  eng-sdn.csv  mar-eng.csv  msa-dza.csv  msa-mar.csv  msa-sau.csv  msa-syr.csv  pse-msa.csv  sau-msa.csv  sdn-msa.csv  syr-msa.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-4}
  prepare-instruct ${input_file} ${output_file}
done


training_dir="saudial"
training_files=(eng-sau-new.csv msa-sau-new.csv sau-eng-new.csv sau-msa-new.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-8}
  prepare-instruct ${input_file} ${output_file}
done


training_dir="ufal"
training_files=(eng-syr-new.csv msa-syr-new.csv syr-eng-new.csv syr-msa-new.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-8}
  prepare-instruct ${input_file} ${output_file}
done

training_dir="flores-dev"
training_files=(egy-eng.csv  egy-msa.csv  eng-egy.csv  eng-mar.csv eng-pse.csv eng-sau.csv eng-syr.csv mar-eng.csv mar-msa.csv msa-egy.csv msa-mar.csv msa-pse.csv msa-sau.csv msa-syr.csv pse-eng.csv pse-msa.csv sau-eng.csv sau-msa.csv syr-eng.csv syr-msa.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-4}
  prepare-instruct ${input_file} ${output_file}
done

training_dir="madar26-dev"
training_files=(dza-eng.csv  egy-msa.csv  eng-mar.csv  eng-sdn.csv  mar-msa.csv  msa-mar.csv  msa-sdn.csv  pse-msa.csv	sau-msa.csv  syr-eng.csv dza-msa.csv eng-dza.csv  eng-pse.csv  eng-syr.csv  msa-dza.csv  msa-pse.csv  msa-syr.csv sdn-eng.csv  syr-msa.csv egy-eng.csv eng-egy.csv  eng-sau.csv  mar-eng.csv  msa-egy.csv  msa-sau.csv  pse-eng.csv  sau-eng.csv sdn-msa.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-4}
  prepare-instruct ${input_file} ${output_file}
done

