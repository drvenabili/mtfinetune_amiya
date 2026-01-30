# learning rate
config_file="${1}"
yaml_name=$(basename ${config_file})

lrs=(2e-5 3e-5 5e-5 6e-5)
lora_parameters=("16 32")
for lr in "${lrs[@]}"
do
  for pair in "${lora_parameters[@]}"
  do
    read -r LORA_R LORA_ALPHA <<< "$pair"
    bash ./scripts/bash/finetune/instruct.sh 10 ./scripts/slurm/finetune/instruct.sh ${config_file} afterany \
      -o training.learning_rate="${lr}" \
      -o output_dir=runs/${yaml_name::-5}-lr_${lr}_r_${LORA_R}_${LORA_ALPHA} \
      -o log_file=runs/${yaml_name::-5}-lr_${lr}_r_${LORA_R}_${LORA_ALPHA}/train.log \
      -o log_jsonl=runs/${yaml_name::-5}-lr_${lr}_r_${LORA_R}_${LORA_ALPHA}/metrics.jsonl \
      -o lora_quant.lora_r="${LORA_R}" \
      -o lora_quant.lora_alpha="${LORA_ALPHA}"
  done
done
