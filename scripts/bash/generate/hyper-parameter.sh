model=${1}
output_directory=${2}
method=${3}

#temperatures=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
# llama
#top_ps=(0.1 0.3 0.6 0.9 1)
#temperatures=(0.3 0.6 0.9 1)
temperatures=(0.3 0.6 0.9 1)
top_ps=(0.1 0.3 0.6 0.9 1)

for temp in "${temperatures[@]}"
do
  for top_p in "${top_ps[@]}"
  do
    bash ./scripts/bash/generate_from_test.sh ${model} ${output_directory}_${top_p}_${temp} "base" 512 ${temp} ${top_p}
  done
done

