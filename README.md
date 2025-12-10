# Requirements

Python3.10+ and Pytorch2.2

```
uv sync
uv run scripts/python/finetune/finetune.py --help
```
## Installig this repository

1. Load modules to have python and CUDA 

$ ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0

2. Install uv

$ curl -LsSf https://astral.sh/uv/install.sh | sh

3. Create an environment using uv

$ uv venv .env

4. Install pytorch (for Linux and CUDA12.8)

$ uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

5. Install huggingface 

$ uv pip install transformers

6. Health test

$ ./scripts/bash/health.sh

Expected output:

There is a gpu available!

```
## Scripts

The `scripts` folder contains three main directories:

- **python** — Python source code  
- **slurm** — SLURM scripts used to run the Python code on the server  
- **bash** — Bash scripts for submitting SLURM jobs in batch with different configurations (e.g., for hyperparameter search)
```

## TODO
- choisir base model
- remplacer dataset name
- écrire du code de test?
