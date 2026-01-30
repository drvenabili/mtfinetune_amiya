# learning rate
config_file="${1}"
yaml_name=$(basename ${config_file})

lrs=(2e-5 3e-5 5e-5 6e-5)
for lr in "${lrs[@]}"
do
  bash ./scripts/bash/finetune/instruct.sh 10 ./scripts/slurm/finetune/instruct.sh ${config_file} afterany \
    -o training.learning_rate="${lr}" \
    -o output_dir=runs/${yaml_name::-5}-lr_${lr} \
    -o log_file=runs/${yaml_name::-5}-lr_${lr}/train.log \
    -o log_jsonl=runs/${yaml_name::-5}-lr_${lr}/metrics.jsonl
done
