

# Directory for MT data
INPUT_DIR="./training_data/instruct_prompts/bi"
OUTPUT_DIR="./training_data/tokenized/bi"

tokenizer="${1}/config"

mkdir -p "${OUTPUT_DIR}/${tokenizer}"
function prepare-instruct() {
  input_file="${1}"
  output_file="${2}"
 
  sbatch ./scripts/slurm/preprocess/prepare-instruct.sh "${input_file}" "${tokenizer}" "${output_file}" 2048 " "
}

##
# atlas  casablanca  doda  joda  madar26-train  saudial  ufal

training_dir="atlas"
training_files=(eng-mar-new.csv mar-eng-new.csv)

for data in "${training_files[@]}"
do
  input_file="${INPUT_DIR}"/${training_dir}/${data}
  output_file="${OUTPUT_DIR}"/${tokenizer}/${training_dir}/${data::-8}
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

