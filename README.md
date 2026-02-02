# Aladdin-FTI @ AMIYA
## Three Wishes for Arabic NLP: Fidelity, Diglossia, and Multidialectal Generation

**Authors**:  
Jonathan Mutal<sup>×</sup>, Perla Al Almaoui<sup>×</sup>, Simon Hengchen<sup>×</sup><sup>+</sup>, and Pierrette Bouillon<sup>×</sup>

**Affiliations**:  
<sup>×</sup>TIM, University of Geneva  
<sup>+</sup>iguanodon.ai  

**Paper**:  
[Aladdin-FTI @ AMIYA: Large Three Wishes for Arabic NLP: Fidelity, Diglossia, and Multidialectal Generation](https://aclanthology.org/####)

**Model**:  
[🤗 Available on Hugging Face](https://huggingface.co/jonathanmutal/XXX)

# Description

Utilities to **fine-tune, generate, and evaluate** causal LLMs for **machine translation** (MT) and **dialect** experiments.

This repository is organized around:
- a **Typer CLI with YAML configuration file** (LoRA + optional quantization, custom eval hooks) (`scripts/python/finetune/instruct.py`)
- **batch generation**, **automatic evaluation** (ChrF++, SpBLEU, + dialect-ID based fidelity scores), and **MBR ranking**

---

## Features

- **Fine-tuning**
  - Full fine-tuning (TRL / SFTTrainer) or **LoRA/PEFT**
  - YAML configs for reproducible runs (training hyperparams, dataset lists, metrics, checkpoint selection)
- **Generation**
  - Single prompt generation
  - Batch generation from a text file
- **Evaluation**
  - MT metrics: **ChrF++** and **SpBLEU** (SacreBLEU)
  - Dialect/fidelity scoring via ADI-family models (ALDI/NADI) + fastText language-ID helpers
- **Ranking**
  - MBR-style ranking script (`scripts/python/rank/mbr_rank.py`)
- **HPC-friendly**
  - SLURM scripts (`scripts/slurm/**`) + bash wrappers (`scripts/bash/**`)

---

## Repository layout

```
.
├── configs.py                       # Small helper configs (LoRA, SFTConfig, prompt templates)
├── configs/                         # YAML training configs (instruct pipeline)
│   └── instruct/*.yaml
├── scripts/
│   ├── python/
│   │   ├── finetune/                # fine-tuning entrypoints
│   │   ├── generate/                # generation entrypoints
│   │   ├── evaluate/                # evaluation (ChrF/SpBLEU + dialect/fidelity)
│   │   ├── preprocess/              # dataset preparation helpers
│   │   ├── rank/                    # MBR ranking
│   ├── slurm/                       # SLURM job scripts (call the python entrypoints)
│   └── bash/                        # wrappers to submit batches
└── uv.lock                          # dependency lockfile (uv)
```
---

## Requirements

- **Python**: 3.10+
- **PyTorch**: 2.x
- CUDA GPU recommended

Key libraries (see `uv.lock`): `transformers`, `datasets`, `trl`, `peft`, `accelerate`, `sacrebleu`, `fasttext`, `huggingface_hub`, `typer`, `pyyaml` (for YAML pipeline).

---

## Setup (recommended: `uv`)

> The original README targets an HPC module environment. Adapt the module load lines to your cluster.

### 1) Load CUDA + Python (example)

```bash
ml load GCCcore/11.3.0 Python/3.10.4 CUDA/12.8.0
```

### 2) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3) Create a venv and install deps

```bash
uv venv .env
source .env/bin/activate
uv sync
```

### 4) Install PyTorch (example for Linux + CUDA 12.8)

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 5) Sanity check (GPU)

```bash
./scripts/bash/health.sh
# or
uv run scripts/python/health.py
```

---


## Quickstart

### using the YAML file to fine-tune (recommended for experiments)

The `instruct.py` pipeline reads a YAML config (see `configs/instruct/*.yaml`) and supports:
- LoRA + optional quantization
- multiple datasets (train/eval) from disk (`datasets.load_from_disk`)
- structured logging (log file + metrics jsonl)
- evaluation hooks + “best checkpoint” tracking

Example:

```bash
uv run scripts/python/finetune/instruct.py --help
uv run scripts/python/finetune/instruct.py train configs/instruct/mt-fidelity-all-small-template-llama-8B.yaml
```

---

## Generation

### Single prompt

```bash
uv run scripts/python/generate/generate.py generate   "Translate: Bonjour"   --model-path SmolLM3-3B-aladdinFTI-sft-trl   --method trl
```

### Batch generation (one prompt per line)

```bash
uv run scripts/python/generate/generate.py generate-batch   data/prompts.txt   --model-path SmolLM3-3B-aladdinFTI-sft-trl   --method trl   --batch-size 16   --max-new-tokens 256
```

---

## Evaluation

The evaluator supports `.txt`, `.out`, and `.csv` inputs:

- MT metrics (SacreBLEU):
  - **ChrF++** 
  - **SpBLEU** (optional) (`BLEU(tokenize="flores200")`)
- Fidelity / dialect scores:
  - ADI-family classification models (ALDI/NADI) + mapping utilities in `scripts/python/evaluate/maps.py`

Example (see help for exact flags):

```bash
uv run scripts/python/evaluate/evaluator.py --help
```

---

## MBR ranking

Use `mbr_rank.py` to rank candidate generations using reference-free/metric-based selection.

```bash
uv run scripts/python/rank/mbr_rank.py --help
```

Cluster-friendly wrappers exist in:
- `scripts/bash/rank/*`
- `scripts/slurm/rank/mbr_rank.sh`

---

## Running on SLURM

You can submit jobs directly with the provided SLURM scripts.

YAML training (example):

```bash
sbatch scripts/slurm/finetune/instruct.sh configs/instruct/mt-fidelity-all-small-template-llama-8B.yaml
```

Batch generation:

```bash
sbatch scripts/slurm/generate/generate-batch.sh path/to/prompts.txt SmolLM3-3B-aladdinFTI-sft-trl trl
```

