# example of simple prompt
prompt="Give me a brief explanation of gravity in simple terms."
model=HuggingFaceTB/SmolLM3-3B
method="base"

# We recommend setting temperature=0.6 and top_p=0.95 in the sampling parameters.
max_new_tokens=1024
temperature=0.6
top_p=0.95

sbatch ./scripts/slurm/generate/generate.sh "${prompt}" ${model} ${method} ${max_new_tokens} ${temperature} ${top_p}

