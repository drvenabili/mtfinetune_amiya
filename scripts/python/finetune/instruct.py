#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import math
import re
import signal
import time
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import typer
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)

# ---- optional deps ----
try:
    import yaml  # PyYAML
except Exception:
    yaml = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    from sacrebleu.metrics import CHRF
except Exception:
    CHRF = None

### add the token for huggingface
HF_TOKEN = ""
app = typer.Typer(add_completion=False, help="SFT training with YAML config, dataset-specific metrics and val splits.")


# -----------------------------------------------------------------------------
# Debugging
# -----------------------------------------------------------------------------
def label_coverage(ds, k=50):
    ds2 = ds.select(range(min(k, len(ds))))
    cov = []
    for ex in ds2:
        labels = ex["labels"]
        keep = sum(1 for t in labels if t != -100)
        cov.append(keep / max(1, len(labels)))
    return sum(cov)/len(cov)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(log_file: Optional[Path], level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("sft_yaml")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def append_jsonl(path: Optional[Path], payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["ts"] = datetime.utcnow().isoformat() + "Z"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# SLURM-friendly signal handling
# -----------------------------------------------------------------------------
_should_stop = {"flag": False, "signal": None}


def _handle_stop(signum, frame):
    _should_stop["flag"] = True
    _should_stop["signal"] = signum
    print(f"[{time.strftime('%F %T')}] Received signal {signum} in PID={os.getpid()}", flush=True)


signal.signal(signal.SIGUSR1, _handle_stop)
signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)


class SaveOnSignalCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if _should_stop["flag"]:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        # called during gradient accumulation in recent transformers
        if _should_stop["flag"]:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if _should_stop["signal"] is not None:
            typer.echo(f"Training ended due to signal: {_should_stop['signal']}")
        return control


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, logger: logging.Logger, log_jsonl: Optional[Path]):
        self.logger = logger
        self.log_jsonl = log_jsonl

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        bits = []
        for k in ("loss", "eval_loss", "learning_rate", "grad_norm", "epoch"):
            if k in logs:
                v = logs[k]
                bits.append(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}")
        if bits:
            self.logger.info(f"step={state.global_step} | " + " | ".join(bits))
        append_jsonl(self.log_jsonl, {"step": state.global_step, **logs})


# -----------------------------------------------------------------------------
# Collator
# -----------------------------------------------------------------------------
@dataclass
class DataCollatorForCausalLMWithLabels:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    ## debug
    debug: bool = True
    debug_every: int = 200
    _batch_count: int = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        labels = [f["labels"] for f in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        padded_labels: List[List[int]] = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        # ------------------------------------------------------------------
        # DEBUG: print one random example from this batch
        # ------------------------------------------------------------------
        if self.debug and self._batch_count % self.debug_every == 0:
            idx = random.randint(0, len(input_ids) - 1)

            inp = batch["input_ids"][idx].tolist()
            lab = batch["labels"][idx].tolist()

            input_text = self.tokenizer.decode(inp, skip_special_tokens=False)
            label_ids = [t for t in lab if t != -100]
            label_text = self.tokenizer.decode(label_ids, skip_special_tokens=False)

            coverage = len(label_ids) / max(1, len(inp))

            print("\n" + "=" * 80)
            print(f"Collator debug | batch={self._batch_count}")
            print(f"Label coverage: {coverage:.3f}")
            print("\n[INPUT]")
            print(input_text[:1500])
            print("\n[LABELS – supervised tokens only]")
            print(label_text[:1500])
            print("=" * 80 + "\n")
        return batch


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------
def get_generation_suffix_ids(tokenizer) -> list[int]:
    """
    Returns the token IDs appended by apply_chat_template(..., add_generation_prompt=True)
    compared to add_generation_prompt=False for a dummy user message.
    """
    dummy = [{"role": "user", "content": "__DUMMY__"}]

    with_gen = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
    no_gen  = tokenizer.apply_chat_template(dummy, tokenize=False, add_generation_prompt=False)

    if not with_gen.startswith(no_gen):
        # Some templates may differ in more complex ways; fallback to a safer method:
        # Try to find the last occurrence of the dummy and take everything after it.
        idx = with_gen.rfind("__DUMMY__")
        if idx == -1:
            raise RuntimeError("Could not infer chat-template generation suffix.")
        suffix_text = with_gen[idx + len("__DUMMY__") :]
    else:
        suffix_text = with_gen[len(no_gen):]

    suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
    if len(suffix_ids) == 0:
        raise RuntimeError("Inferred generation suffix is empty; check chat template.")
    return suffix_ids

def print_random_training_examples_with_mask(
    ds: Dataset,
    tokenizer,
    n: int = 2,
):
    print("\n" + "=" * 80)
    print("Training examples with label mask visualization")
    print("=" * 80)

    idxs = random.sample(range(len(ds)), k=min(n, len(ds)))

    for idx in idxs:
        ex = ds[idx]
        input_ids = ex["input_ids"]
        labels = ex["labels"]

        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        print(f"\n--- idx={idx} ---")
        for tok, lab in zip(tokens, labels):
            if lab == -100:
                print(tok, end=" ")
            else:
                print(f"[{tok}]", end=" ")
        print("\n")

    print("=" * 80 + "\n")

REQUIRED_COLS = ("input_ids", "attention_mask", "labels")

def _load_dataset_any(path: Path, split: str = "train") -> Dataset:
    obj = load_from_disk(str(path))
    if isinstance(obj, Dataset):
        return obj
    if isinstance(obj, DatasetDict):
        if split in obj:
            return obj[split]
        # fallback: first split
        first = next(iter(obj.keys()))
        return obj[first]
    raise typer.BadParameter(f"Unsupported dataset object at: {path}")

def _validate_schema(ds: Dataset, name: str) -> None:
    cols = set(ds.column_names)
    missing = [c for c in REQUIRED_COLS if c not in cols]
    if missing:
        raise typer.BadParameter(f"{name} missing columns {missing}. Found: {sorted(cols)}")


def _repeat_dataset(ds: Dataset, times: int) -> Dataset:
    if times <= 1:
        return ds
    return concatenate_datasets([ds] * times)


def _subsample(ds: Dataset, *, frac: Optional[float], n: Optional[int], seed: int) -> Dataset:
    if n is not None and n > 0:
        n = min(n, len(ds))
        return ds.select(range(n))
    if frac is not None:
        if not (0.0 < frac <= 1.0):
            raise typer.BadParameter(f"sample_frac must be in (0,1], got {frac}")
        if frac >= 1.0:
            return ds
        k = max(1, int(round(len(ds) * frac)))
        # shuffle then take first k for determinism
        return ds.shuffle(seed=seed).select(range(k))
    return ds


def _list_checkpoints(output_dir: Path) -> List[Tuple[int, Path]]:
    if not output_dir.exists():
        return []
    ckpts: List[Tuple[int, Path]] = []
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            m = re.match(r"^checkpoint-(\d+)$", p.name)
            if m:
                ckpts.append((int(m.group(1)), p))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    ckpts = _list_checkpoints(output_dir)
    return ckpts[-1][1] if ckpts else None


def resolve_resume_checkpoint(output_dir: Path, resume_from_checkpoint: Optional[Path], auto_resume: bool) -> Optional[Path]:
    if resume_from_checkpoint is not None:
        if not resume_from_checkpoint.exists():
            raise typer.BadParameter(f"resume_from_checkpoint not found: {resume_from_checkpoint}")
        return resume_from_checkpoint

    marker = output_dir / "latest_checkpoint.txt"
    if auto_resume and marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip())
        if p.exists():
            return p

    if auto_resume:
        return find_latest_checkpoint(output_dir)
    return None

### This is usually for multi-turn conversation. We can package many templtes
### in one row
def maybe_pack_dataset(ds: Dataset, max_length: int) -> Dataset:
    def _pack_batch(batch):
        packed = {"input_ids": [], "attention_mask": [], "labels": []}
        cur_ids, cur_attn, cur_lbls = [], [], []
        cur_len = 0

        for ids, attn, lbls in zip(batch["input_ids"], batch["attention_mask"], batch["labels"]):
            if not ids:
                continue
            if cur_len + len(ids) > max_length and cur_len > 0:
                packed["input_ids"].append(cur_ids[:max_length])
                packed["attention_mask"].append(cur_attn[:max_length])
                packed["labels"].append(cur_lbls[:max_length])
                cur_ids, cur_attn, cur_lbls = [], [], []
                cur_len = 0

            cur_ids.extend(ids)
            cur_attn.extend(attn)
            cur_lbls.extend(lbls)
            cur_len += len(ids)

        if cur_len > 0:
            packed["input_ids"].append(cur_ids[:max_length])
            packed["attention_mask"].append(cur_attn[:max_length])
            packed["labels"].append(cur_lbls[:max_length])

        return packed

    return ds.map(_pack_batch, batched=True, batch_size=256, desc="Packing sequences")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_chrfpp_on_eval_subset(
    model,
    tokenizer,
    eval_ds: Dataset,
    n: int,
    gen_max_new_tokens: int,
    gen_temperature: float,
    gen_top_p: float,
) -> float:
    if n <= 0:
        return float("nan")
    if CHRF is None:
        raise RuntimeError("sacrebleu is not installed. Install with: pip install sacrebleu")

    n = min(n, len(eval_ds))
    subset = eval_ds.select(range(n))

    metric = CHRF(word_order=2)  # chrF++
    hyps: List[str] = []
    refs: List[str] = []

    model.eval()
    device = next(model.parameters()).device

    for ex in subset:
        input_ids: List[int] = ex["input_ids"]
        labels: List[int] = ex["labels"]

        # boundary: first position where labels != -100
        cut = 0
        while cut < len(labels) and labels[cut] == -100:
            cut += 1

        prompt_ids = input_ids[:cut]
        ref_ids = [t for t in labels[cut:] if t != -100]
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True).strip()
        print(f"ref_text: {ref_text}")
        if not ref_text:
            continue

        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        gen = model.generate(
            input_ids=prompt,
            max_new_tokens=gen_max_new_tokens,
            do_sample=(gen_temperature > 0.0),
            temperature=gen_temperature if gen_temperature > 0.0 else None,
            top_p=gen_top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id
        )
        gen_ids = gen[0].tolist()
        cont_ids = gen_ids[len(prompt_ids) :]
        hyp_text = tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
        print(f"hypothesis: {hyp_text}")
        hyps.append(hyp_text)
        refs.append(ref_text)

    if not refs:
        return float("nan")
    return float(metric.corpus_score(hyps, [refs]).score)


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------
def _ensure_deps_for_lora_or_kbit(use_lora: bool, quant_bits: int) -> None:
    if (use_lora or quant_bits in (4, 8)) and (LoraConfig is None or get_peft_model is None):
        raise RuntimeError("peft is required (pip install peft)")
    if quant_bits in (4, 8):
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            raise RuntimeError("bitsandbytes is required (pip install bitsandbytes)")


def _build_bnb_config(quant_bits: int) -> Optional[BitsAndBytesConfig]:
    if quant_bits == 0:
        return None
    if quant_bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError("quant_bits must be 0, 4, or 8")


def _infer_torch_dtype(dtype: str) -> torch.dtype:
    d = dtype.lower().strip()
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16"):
        return torch.float16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError("dtype must be bf16, fp16, or fp32")


# -----------------------------------------------------------------------------
# YAML config schema (informal)
# -----------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get(d: dict, key: str, default=None):
    return d[key] if key in d else default


# -----------------------------------------------------------------------------
# Dataset plan builder
# -----------------------------------------------------------------------------
def build_datasets_from_config(cfg: dict,
                               seed: int,
                               logger: logging.Logger) -> Tuple[Dataset, Dict[str, dict]]:
    """
    Returns:
      - train_ds (concatenated)
      - eval_groups: dict[name] = {
            "dataset": Dataset,
            "metrics": set[str],   # {"loss","ppl","chrfpp"}
            "chrf_eval_n": int,
            "metric_key_prefix": str,  # used by trainer.evaluate()
        }

    Supports:
      - role: "train" or "eval"
      - per-dataset val_ratio for train datasets: split part into its own eval group
      - per-dataset sampling for eval groups: sample_frac or sample_n
      - per-dataset repeat/upsample: repeat
      - per-dataset split (if DatasetDict): split
      - per-eval-group metrics: metrics list
    """
    ds_cfgs: List[dict] = get(cfg, "datasets", [])
    if not ds_cfgs:
        raise typer.BadParameter("Config must contain a non-empty 'datasets' list.")

    train_parts: List[Dataset] = []
    eval_groups: Dict[str, dict] = {}

    for i, dc in enumerate(ds_cfgs):
        name = str(get(dc, "name", f"ds{i}"))
        role = str(get(dc, "role", "train")).lower()
        path = Path(str(dc["path"])) if "path" in dc else None
        if path is None:
            raise typer.BadParameter(f"datasets[{i}] missing 'path'")
        if not path.exists():
            raise typer.BadParameter(f"Dataset not found: {path}")

        split = str(get(dc, "split", "train"))
        repeat = int(get(dc, "repeat", 1))
        sample_frac = get(dc, "sample_frac", None)
        sample_n = get(dc, "sample_n", None)
        val_ratio = float(get(dc, "val_ratio", 0.0))  # only used when role=train

        ds = _load_dataset_any(path, split=split)
        _validate_schema(ds, f"{role}:{name}:{path}")

        if repeat > 1:
            ds = _repeat_dataset(ds, repeat)

        if role == "train":
            # if dataset wants its own val split, split it BEFORE concatenating
            if val_ratio > 0.0:
                val_ratio = int(val_ratio) if val_ratio > 1 else val_ratio
                split_dd = ds.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
                ds_train = split_dd["train"]
                ds_eval = split_dd["test"]

                train_parts.append(ds_train)
                logger.info(f"[{name}] train split: train={len(ds_train)} eval={len(ds_eval)} (val_ratio={val_ratio})")

                # Create an eval group for this dataset split
                ev_name = f"{name}_val"
                # sampling on eval split (if desired)
                ds_eval = _subsample(ds_eval, frac=sample_frac, n=sample_n, seed=seed)
                metrics = set(m.lower() for m in (get(dc, "metrics", ["loss"]) or ["loss"]))
                eval_groups[ev_name] = {
                    "dataset": ds_eval,
                    "metrics": metrics,
                    "chrf_eval_n": int(get(dc, "chrf_eval_n", get(cfg.get("metrics", {}), "chrf_eval_n", 64))),
                    "metric_key_prefix": ev_name,
                }
                logger.info(f"[{name}] train: n={len(ds_train)}")
                logger.info(f"[{name}] eval: n={len(ds_eval)} metrics={sorted(metrics)}")
            else:
                train_parts.append(ds)
                logger.info(f"[{name}] train: n={len(ds)}")
        elif role == "eval":
            # eval dataset can be sampled independently
            ds = _subsample(ds, frac=sample_frac, n=sample_n, seed=seed)
            metrics = set(m.lower() for m in (get(dc, "metrics", ["loss"]) or ["loss"]))
            eval_groups[name] = {
                "dataset": ds,
                "metrics": metrics,
                "chrf_eval_n": int(get(dc, "chrf_eval_n", get(cfg.get("metrics", {}), "chrf_eval_n", 64))),
                "metric_key_prefix": name,
            }
            logger.info(f"[{name}] eval: n={len(ds)} metrics={sorted(metrics)}")
        else:
            raise typer.BadParameter(f"datasets[{i}] invalid role: {role!r}")

    if not train_parts:
        raise typer.BadParameter("No training data produced. Add at least one dataset with role: train")

    train_ds = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
    train_ds = train_ds.shuffle(seed=seed)

    return train_ds, eval_groups


def choose_trainer_eval_dataset(eval_groups: Dict[str, dict],
                                prefer: Sequence[str]) -> Optional[Tuple[str, Dataset]]:
    """
    Trainer supports only ONE eval_dataset for its built-in evaluation loop.
    We choose one group to attach to Trainer (for eval_loss logging etc.).
    But we still compute metrics for other eval groups via a callback.
    """
    if not eval_groups:
        return None
    for name in prefer:
        if name in eval_groups:
            return name, eval_groups[name]["dataset"]
    # fallback: first
    first = next(iter(eval_groups.keys()))
    return first, eval_groups[first]["dataset"]


# -----------------------------------------------------------------------------
# Dataset-specific evaluation callback
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_chrfpp_on_eval_subset_batched(
    model,
    tokenizer,
    eval_ds: Dataset,
    n: int,
    gen_max_new_tokens: int,
    gen_temperature: float,
    gen_top_p: float,
    batch_size: int = 8,
    debug_print: bool = False,
) -> float:
    if n <= 0:
        return float("nan")
    if CHRF is None:
        raise RuntimeError("sacrebleu is not installed. Install with: pip install sacrebleu")

    n = min(n, len(eval_ds))
    subset = eval_ds.select(range(n))

    metric = CHRF(word_order=2)  # chrF++
    hyps: List[str] = []
    refs: List[str] = []

    model.eval()
    device = next(model.parameters()).device

    # get the prefix when generating
    suffix_ids = get_generation_suffix_ids(tokenizer)

    # ---- Build prompt/ref lists (same logic as your per-example loop) ----
    prompts: List[List[int]] = []
    prompt_lens: List[int] = []
    ref_texts: List[str] = []
    ref_tok_lens: List[int] = []

    for ex in subset:
        input_ids: List[int] = ex["input_ids"]
        labels: List[int] = ex["labels"]

        # boundary: first position where labels != -100
        cut = 0
        while cut < len(labels) and labels[cut] == -100:
            cut += 1

        prompt_ids = input_ids[:cut]
        # Ensure the prompt ends with the assistant generation prefix
        if len(prompt_ids) < len(suffix_ids) or prompt_ids[-len(suffix_ids):] != suffix_ids:
            prompt_ids = prompt_ids + suffix_ids
        ref_ids = [t for t in labels[cut:] if t != -100]
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True).strip()

        if not ref_text or not prompt_ids:
            continue

        prompts.append(prompt_ids)
        prompt_lens.append(len(prompt_ids))
        ref_texts.append(ref_text)
        ref_tok_lens.append(len(ref_ids))

    if not ref_texts:
        return float("nan")

    hyp_lens_tok: List[int] = []
    ref_lens_tok: List[int] = []
    # ---- Batched generate ----
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_lens = prompt_lens[i : i + batch_size]
        batch_refs = ref_texts[i : i + batch_size]
        batch_ref_tok_lens = ref_tok_lens[i : i + batch_size]
        # Pad prompts to a rectangular tensor (like your single prompt tensor)
        padded = tokenizer.pad(
            {"input_ids": batch_prompts},
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )
        input_ids = padded["input_ids"].to(device)
        attention_mask = padded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_max_new_tokens,
            do_sample=(gen_temperature > 0.0),
            temperature=gen_temperature if gen_temperature > 0.0 else None,
            top_p=gen_top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id
        )

        input_seq_len = input_ids.shape[1]  # padded length (same for whole batch)
        # Decode per-sample continuation (same as your cont_ids slicing)
        gen = gen.detach().cpu()
        for j in range(gen.shape[0]):
            gen_ids = gen[j].tolist()
            cont_ids = gen_ids[input_seq_len:]
            hyp_text = tokenizer.decode(cont_ids, skip_special_tokens=True).strip()

            if debug_print:
                print(f"input_text: {tokenizer.decode(input_ids[j].detach().cpu().tolist(), skip_special_tokens=False)}")
                print(f"with_attention: {tokenizer.decode((input_ids[j]*attention_mask[j]).detach().cpu().tolist(), skip_special_tokens=False)}")
                print(f"ref_text: {ref_text}")
                print(f"hypothesis: {hyp_text}")

            hyps.append(hyp_text)
            refs.append(batch_refs[j])
            hyp_lens_tok.append(len(cont_ids))              # token length from vectors
            ref_lens_tok.append(batch_ref_tok_lens[j])      # token length from vectors

    if not refs:
        return float("nan")
    if debug_print:
        print("hyps:", hyps)
        print("refs:", refs)
    chrf = float(metric.corpus_score(hyps, [refs]).score)
    avg_hyp_len = float(sum(hyp_lens_tok) / len(hyps))
    avg_ref_len = float(sum(ref_lens_tok) / len(ref_lens_tok))
    return float(metric.corpus_score(hyps, [refs]).score), avg_hyp_len, avg_ref_len

class BestMetricCheckpointCallback(TrainerCallback):
    """
    Keeps ONE best checkpoint per metric (e.g., best chrF++, best ppl).
    On each evaluation:
      - finds the latest 'checkpoint-XXXX' produced by Trainer
      - if metric improved, copies that checkpoint to output_dir/best_<metric_name>/

    Notes:
      - For ppl/loss: smaller is better (mode='min')
      - For chrF++: larger is better (mode='max')
    """
    def __init__(
        self,
        *,
        output_dir: Path,
        metric_specs: Dict[str, str],  # metric_key -> "min"|"max"
        logger: logging.Logger,
    ):
        self.output_dir = Path(output_dir)
        self.metric_specs = metric_specs
        self.logger = logger
        self.best: Dict[str, float] = {}          # metric_key -> best value
        self.best_step: Dict[str, int] = {}       # metric_key -> step
        self.pending_steps: dict[str, int] = {}

        self.state_path = self.output_dir / "best_metrics_state.json"
        self._loaded = False

    # -------- persistence --------
    def _load_state(self):
        if self._loaded:
            return
        self._loaded = True
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.best = {k: float(v) for k, v in data.get("best", {}).items()}
                self.best_step = {k: int(v) for k, v in data.get("best_step", {}).items()}
                # pending_steps is ephemeral; safer to reset after restart
                self.pending_steps = {}
                self.logger.info(f"Loaded best-metrics state from {self.state_path}")
            except Exception as e:
                self.logger.warning(f"Could not load {self.state_path}: {e}")

    def _save_state(self):
        tmp = self.state_path.with_suffix(".json.tmp")
        payload = {"best": self.best, "best_step": self.best_step}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def _is_better(self, key: str, val: float) -> bool:
        mode = self.metric_specs[key]
        if key not in self.best:
            return True
        if mode == "min":
            return val < self.best[key]
        if mode == "max":
            return val > self.best[key]
        raise ValueError(f"Unknown mode for {key}: {mode}")

    def _copy_checkpoint_single(self, src_ckpt: Path, dst_dir: Path) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in dst_dir.glob("checkpoint-*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
        shutil.copytree(src_ckpt, dst_dir / src_ckpt.name)

        self.logger.info(f"Saved best checkpoint copy: {src_ckpt.name} -> {dst_dir}")

    # Load state as soon as we know args.output_dir (covers fresh + resumed runs)
    def on_train_begin(self, args, state, control, **kwargs):
        self.output_dir = Path(args.output_dir)
        self.state_path = self.output_dir / "best_metrics_state.json"
        self._load_state()
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._load_state()
        if not metrics:
            return control

        step = int(state.global_step)

        for key, mode in self.metric_specs.items():
            if key not in metrics:
                continue
            val = float(metrics[key])
            if math.isnan(val):
                continue

            if self._is_better(key, val):
                self.best[key] = val
                self.pending_steps[key] = step
                control.should_save = True  # ensure checkpoint-step will exist
                self.logger.info(f"New best {key} ({mode}) = {val:.6g} at step={step}")
                self._save_state()

        return control

    def on_save(self, args, state, control, **kwargs):
        self._load_state()
        step = int(state.global_step)
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not ckpt_dir.exists():
            return control

        for key, pending_step in list(self.pending_steps.items()):
            if pending_step != step:
                continue

            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
            dst_dir = Path(args.output_dir) / f"best_{safe}"

            self._copy_checkpoint_single(ckpt_dir, dst_dir)
            self.logger.info(f"Best-per-metric saved for {key} from checkpoint-{step}")

            del self.pending_steps[key]

        return control


class MultiEvalCallback(TrainerCallback):
    """
    At each evaluation (and/or end), compute metrics on each eval group.
    - For loss/ppl: uses trainer.evaluate(eval_dataset=..., metric_key_prefix=group_name)
    - For chrF++: uses generation on that group's dataset subset
    """
    def __init__(
        self,
        *,
        trainer: Trainer,
        tokenizer,
        logger: logging.Logger,
        log_jsonl: Optional[Path],
        eval_groups: Dict[str, dict],
        do_on_steps: bool,
        do_on_final: bool,
        gen_max_new_tokens: int,
        gen_temperature: float,
        gen_top_p: float,
        debug: bool,
    ):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.logger = logger
        self.log_jsonl = log_jsonl
        self.eval_groups = eval_groups
        self.do_on_steps = do_on_steps
        self.do_on_final = do_on_final
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.debug = debug

    @torch.no_grad()
    def _loss_and_ppl(self, ds: Dataset) -> tuple[float, float]:
        model = self.trainer.model

        dataloader = self.trainer.get_eval_dataloader(ds)

        total_loss = 0.0
        total_count = 0

        for batch in dataloader:
            # batch already on device in recent Trainer stacks; if not, uncomment next line:
            # batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            bs = batch["input_ids"].size(0)
            total_loss += float(loss) * bs
            total_count += bs

        mean_loss = total_loss / max(1, total_count)
        try:
            ppl = float(math.exp(mean_loss))
        except OverflowError:
            ppl = float("inf")
        return mean_loss, ppl

    def _run_all(self, step: int) -> Dict[str, float]:
        out_row: Dict[str, Any] = {"step": step}
        self.trainer.model.eval()

        chrf_values: List[float] = []
        loss_values: List[float] = []
        for name, g in self.eval_groups.items():
            ds: Dataset = g["dataset"]
            metrics: set = g["metrics"]
            prefix: str = g["metric_key_prefix"]

            # loss/ppl via Trainer
            if ("loss" in metrics) or ("ppl" in metrics):
                loss, ppl = self._loss_and_ppl(ds)
                # m contains keys like f"{prefix}_eval_loss"
                # log compact
                k_loss = f"{prefix}_eval_loss"
                k_ppl = f"{prefix}_eval_ppl"

                out_row[k_loss] = loss
                out_row[k_ppl]  = ppl

                self.logger.info(f"[{name}] eval_loss = {loss:.6f}")
                self.logger.info(f"[{name}] eval_ppl = {ppl:.2f}")

                # if you still want these in Trainer logs/history:
                self.trainer.log({k_loss: loss, k_ppl: ppl})

                # update general loss
                loss_values.append(loss)

            # chrF++ via generation
            if "chrfpp" in metrics:
                n = int(g.get("chrf_eval_n", 64))
                t0 = time.time()
                score, avg_hyp_len, avg_ref_len = compute_chrfpp_on_eval_subset_batched(
                    model=self.trainer.model,
                    tokenizer=self.tokenizer,
                    eval_ds=ds,
                    n=n,
                    gen_max_new_tokens=self.gen_max_new_tokens,
                    gen_temperature=self.gen_temperature,
                    gen_top_p=self.gen_top_p,
                    debug_print=self.debug
                )
                chrf_values.append(score)
                dt = time.time() - t0
                k = f"{prefix}_eval_chrfpp"
                out_row[k] = score
                self.logger.info(f"[{name}] eval_chrf++ = {score:.3f} | n = {min(n, len(ds))} | took = {dt:.2f}s | avg_hyp_len = {avg_hyp_len:.2f} | avg_ref_len = {avg_ref_len:.2f}")
                self.trainer.log({k: score})

        if chrf_values:
            avg_chrf = sum(chrf_values) / len(chrf_values)
            out_row["eval_chrfpp_avg_macro"] = avg_chrf
            self.logger.info(f"Macro-avg chrF++ = {avg_chrf:.3f}")
        else:
            out_row["eval_chrfpp_avg_macro"] = float("nan")

        if loss_values:
            avg_loss = sum(loss_values) / len(loss_values)
            out_row["eval_loss_avg_macro"] = avg_loss
            self.logger.info(f"Macro-avg loss = {avg_loss:.3f}")
        else:
            out_row["eval_loss_avg_macro"] = float("nan")

        append_jsonl(self.log_jsonl, out_row)
        self.trainer.model.train()
        # Return only scalar metrics (Trainer expects floats)
        return {k: float(v) for k, v in out_row.items() if isinstance(v, (int, float)) and k != "step"}


    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.do_on_steps:
            try:
                extra = self._run_all(step=state.global_step)
                if metrics is not None:
                    metrics.update(extra)   # <-- makes it available to best-model + early stopping
                self.trainer.log(extra)     # optional but nice for logs/history
            except Exception as e:
                self.logger.error(f"Multi-eval on steps failed: {e}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.do_on_final:
            try:
                self._run_all(step=state.global_step)
            except Exception as e:
                self.logger.error(f"Multi-eval on final failed: {e}")
        return control


# -----------------------------------------------------------------------------
# Main entry: YAML-driven
# -----------------------------------------------------------------------------
def deep_set(d: dict, key_path: str, value):
    keys = key_path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def parse_override(s: str):
    # key=value (supports int/float/bool/null/str via JSON-ish parsing)
    if "=" not in s:
        raise typer.BadParameter(f"Invalid override: {s!r} (expected key=value)")
    k, v = s.split("=", 1)
    k = k.strip()
    v = v.strip()
    # try JSON decoding for numbers/bools/null, fallback to raw string
    try:
        v2 = json.loads(v)
    except Exception:
        v2 = v
    return k, v2

@app.command("train")
def train_from_yaml(
    config: Path = typer.Argument(..., exists=True, readable=True),
    override: List[str] = typer.Option([], "--override", "-o", help="Override YAML: key=value (dot paths). Repeatable."),
):
    cfg = load_config(config)

    for s in override:
        k, v = parse_override(s)
        deep_set(cfg, k, v)
    # ---- general ----
    seed = int(get(cfg, "seed", 42))
    set_seed(seed)

    output_dir = Path(str(get(cfg, "output_dir", "sft_out")))
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = get(cfg, "log_file", None)
    log_jsonl = get(cfg, "log_jsonl", None)
    log_level = str(get(cfg, "log_level", "INFO"))
    logger = setup_logger(Path(log_file) if log_file else None, level=log_level)
    log_jsonl_p = Path(log_jsonl) if log_jsonl else None

    debug = bool(get(cfg, "debug", False))
    print(f"debug mode = {debug}")
    # ---- model/tokenizer ----
    model_cfg = cfg.get("model", {})
    model_id = str(model_cfg["model_id"])
    tokenizer_id = str(model_cfg.get("tokenizer_id", model_id))
    dtype = str(model_cfg.get("dtype", "bf16"))
    tf32 = bool(model_cfg.get("tf32", True))
    gradient_checkpointing = bool(model_cfg.get("gradient_checkpointing", True))
    resize_embeddings_if_needed = bool(model_cfg.get("resize_embeddings_if_needed", True))

    # ---- lora/quant ----
    lq = cfg.get("lora_quant", {})
    use_lora = bool(lq.get("use_lora", True))
    quant_bits = int(lq.get("quant_bits", 4))
    lora_r = int(lq.get("lora_r", 16))
    lora_alpha = int(lq.get("lora_alpha", 32))
    lora_dropout = float(lq.get("lora_dropout", 0.05))
    lora_target_modules = str(lq.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"))
    lora_bias = str(lq.get("lora_bias", "none"))

    _ensure_deps_for_lora_or_kbit(use_lora, quant_bits)

    # ---- training ----
    tr = cfg.get("training", {})
    num_train_epochs = float(tr.get("num_train_epochs", 1.0))
    per_device_train_batch_size = int(tr.get("per_device_train_batch_size", 16))
    per_device_eval_batch_size = int(tr.get("per_device_eval_batch_size", 1))
    gradient_accumulation_steps = int(tr.get("gradient_accumulation_steps", 4))
    learning_rate = float(tr.get("learning_rate", 2e-5))
    warmup_ratio = float(tr.get("warmup_ratio", 0.03))
    weight_decay = float(tr.get("weight_decay", 0.0))
    lr_scheduler_type = str(tr.get("lr_scheduler_type", "cosine"))
    max_grad_norm = float(tr.get("max_grad_norm", 1.0))
    optim = str(tr.get("optim", "paged_adamw_8bit"))
    dataloader_num_workers = int(tr.get("dataloader_num_workers", 2))
    group_by_length = bool(tr.get("group_by_length", True))
    logging_steps = int(tr.get("logging_steps", 10))
    save_steps = int(tr.get("save_steps", 200))
    save_total_limit = int(tr.get("save_total_limit", 3))
    eval_steps = int(tr.get("eval_steps", 200))
    report_to = str(tr.get("report_to", "none"))

    auto_resume = bool(tr.get("auto_resume", True))
    resume_from_checkpoint = tr.get("resume_from_checkpoint", None)
    resume_from_checkpoint_p = Path(resume_from_checkpoint) if resume_from_checkpoint else None

    # ---- packing ----
    pack = cfg.get("packing", {})
    pack_sequences = bool(pack.get("pack_sequences", False))
    max_length_hint = int(pack.get("max_length_hint", 2048))

    # ---- metrics decoding settings ----
    met = cfg.get("metrics", {})
    # global defaults for generation-based metric
    gen_max_new_tokens = int(met.get("gen_max_new_tokens", 256))
    gen_temperature = float(met.get("gen_temperature", 0.0))
    gen_top_p = float(met.get("gen_top_p", 1.0))
    multi_eval_on_steps = bool(met.get("on_steps", False))
    multi_eval_on_final = bool(met.get("on_final", True))
    # ---- best checkpoints per metric ----
    best_specs = []
    if isinstance(met, dict):
        best_specs = met.get("best_checkpoints", []) or []

    metric_specs: Dict[str, str] = {}
    if isinstance(best_specs, list):
        for it in best_specs:
            if not isinstance(it, dict):
                continue
            k = str(it.get("key", "")).strip()
            mode = str(it.get("mode", "")).strip().lower()
            if not k or mode not in ("min", "max"):
                continue
            metric_specs[k] = mode

    logger.info(f"config={config}")
    logger.info(f"model_id={model_id}")
    logger.info(f"tokenizer_id={tokenizer_id}")
    logger.info(f"output_dir={output_dir}")

    # ---- datasets ----
    train_ds, eval_groups = build_datasets_from_config(cfg, seed=seed, logger=logger)
    logger.info(f"Train label coverage (avg): {label_coverage(train_ds):.3f}")

    if pack_sequences:
        logger.info(f"Packing enabled (max_length_hint={max_length_hint})")
        train_ds = maybe_pack_dataset(train_ds, max_length=max_length_hint)
        for k in list(eval_groups.keys()):
            eval_groups[k]["dataset"] = maybe_pack_dataset(eval_groups[k]["dataset"], max_length=max_length_hint)

    # Choose one eval group for Trainer's built-in eval loop (optional but useful)
    prefer = met.get("trainer_eval_prefer", [])
    prefer = prefer if isinstance(prefer, list) else []
    chosen = choose_trainer_eval_dataset(eval_groups, prefer=prefer)
    trainer_eval_name, trainer_eval_ds = (chosen if chosen else (None, None))

    if trainer_eval_ds:
        logger.info(f"Eval label coverage (avg): {label_coverage(trainer_eval_ds):.3f}")
    # Determine whether we enable Trainer's evaluation_strategy
    # (we enable it if we have ANY eval group AND multi-eval wants to run on steps or at least keep eval loop)
    do_eval = bool(eval_groups) and (multi_eval_on_steps or multi_eval_on_final)
    if do_eval and trainer_eval_ds is None:
        # should not happen; defensive
        do_eval = False

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        print("Tokenizer pad token is none")
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model ----
    bnb_config = _build_bnb_config(quant_bits)
    torch_dtype = _infer_torch_dtype(dtype)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype if quant_bits == 0 else None,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        token=HF_TOKEN
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if resize_embeddings_if_needed:
        vocab_model = model.get_input_embeddings().num_embeddings
        vocab_tok = len(tokenizer)
        if vocab_tok != vocab_model:
            logger.info(f"Resizing embeddings: model_vocab={vocab_model} -> tokenizer_vocab={vocab_tok}")
            model.resize_token_embeddings(vocab_tok)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if quant_bits in (4, 8):
        if prepare_model_for_kbit_training is None:
            raise RuntimeError("peft is required for k-bit preparation (pip install peft)")
        model = prepare_model_for_kbit_training(model)

    if use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise RuntimeError("peft is required for LoRA (pip install peft)")
        targets = [x.strip() for x in lora_target_modules.split(",") if x.strip()]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)

    use_bf16 = torch_dtype == torch.bfloat16
    use_fp16 = torch_dtype == torch.float16

    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        max_grad_norm=max_grad_norm,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=eval_steps if do_eval else None,
        bf16=use_bf16,
        fp16=use_fp16,
        optim=optim,
        dataloader_num_workers=dataloader_num_workers,
        group_by_length=group_by_length,
        report_to=[] if report_to == "none" else [report_to],
        remove_unused_columns=False,
    )

    collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer,
                                                 debug=debug)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=trainer_eval_ds if do_eval else None,  # one chosen eval dataset to trigger eval loop
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # callbacks
    trainer.add_callback(SaveOnSignalCallback())
    trainer.add_callback(MetricsLoggerCallback(logger=logger, log_jsonl=log_jsonl_p))

    # Multi-eval across groups (loss/ppl/chrfpp per dataset as configured)
    if do_eval:
        trainer.add_callback(
            MultiEvalCallback(
                trainer=trainer,
                tokenizer=tokenizer,
                logger=logger,
                log_jsonl=log_jsonl_p,
                eval_groups=eval_groups,
                do_on_steps=multi_eval_on_steps,
                do_on_final=multi_eval_on_final,
                gen_max_new_tokens=gen_max_new_tokens,
                gen_temperature=gen_temperature,
                gen_top_p=gen_top_p,
                debug=debug,
            )
        )
        logger.info(f"Trainer eval dataset: {trainer_eval_name} (n={len(trainer_eval_ds)})")
    else:
        logger.info("No evaluation configured (no eval groups or metrics disabled).")

    # if we save by best metric:

    if metric_specs:
        trainer.add_callback(
            BestMetricCheckpointCallback(
                output_dir=output_dir,
                metric_specs=metric_specs,
                logger=logger
            )
        )
        logger.info(f"Best-checkpoint tracking enabled for: {metric_specs}")

    # resume
    ckpt = resolve_resume_checkpoint(output_dir, resume_from_checkpoint_p, auto_resume)
    logger.info(f"Resuming from checkpoint: {ckpt}" if ckpt else "Starting from scratch.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable}/{total} = {100*trainable/total:.4f}%")
    # train
    train_result = trainer.train(resume_from_checkpoint=str(ckpt) if ckpt else None)

    # save final
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    latest = find_latest_checkpoint(output_dir)
    if latest is not None:
        (output_dir / "latest_checkpoint.txt").write_text(str(latest), encoding="utf-8")

    # summary
    summary = {
        "config": str(config),
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "output_dir": str(output_dir),
        "seed": seed,
        "train_n": len(train_ds),
        "eval_groups": {
            name: {
                "n": len(g["dataset"]),
                "metrics": sorted(list(g["metrics"])),
                "chrf_eval_n": int(g["chrf_eval_n"]),
            }
            for name, g in eval_groups.items()
        },
        "trainer_eval_group": trainer_eval_name,
        "train_metrics": train_result.metrics,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Done. Output: {output_dir}")

if __name__ == "__main__":
    app()

