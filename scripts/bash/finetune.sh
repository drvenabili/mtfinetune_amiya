#!/bin/bash

method=${1:-trl}
model_name=${2:-HuggingFaceTB/SmolLM3-3B}

uv run scripts/python/finetune.py --method "$method" --model-name "$model_name"
