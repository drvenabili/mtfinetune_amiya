model=${1}
output_directory=${2}
method=${3}
GPU=${4}
temperatures=(0.1 0.3 0.6 0.9 1)
top_ps=(0.1 0.3 0.6 0.9 1)

for temp in "${temperatures[@]}"
do
  for top_p in "${top_ps[@]}"
  do
    bash ./scripts/bash/generate/generate_from_test.sh ${model} ${output_directory}_${top_p}_${temp} ${method} 512 ${temp} ${top_p} 42 ${GPU}
    bash ./scripts/bash/generate/generate_from_dev.sh ${model} ${output_directory}_${top_p}_${temp} ${method} 512 ${temp} ${top_p} 42 ${GPU}
  done
done

