#!/usr/bin/env python3
"""
Like your original script, but:
- prompts-file contains MANY prompts, one per line
- each outputs file contains completions PARALLEL to prompts:
    line i in each output file is the completion for prompt i

For each prompt i:
  candidates = [completion_from_file1[i], completion_from_file2[i], ...]
  compute teacher-forced scores, p_model, ChrF risk, p_risk
  write rows with prompt_id, prompt, etc.
"""

from __future__ import annotations

import csv
import json
import math
import re
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter

import typer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = typer.Typer(add_completion=False)

# ---------------------------
# Utilities: chat template
# ---------------------------

def has_chat_template(tokenizer) -> bool:
    return hasattr(tokenizer, "apply_chat_template") and (
        getattr(tokenizer, "chat_template", None) is not None
        or callable(getattr(tokenizer, "apply_chat_template", None))
    )


def build_full_and_prompt_only_text(
    tokenizer,
    prompt: str,
    completion: str,
    system_prompt: Optional[str] = None,
) -> Tuple[str, str]:
    if not has_chat_template(tokenizer):
        raise ValueError("Tokenizer has no chat template; cannot apply_chat_template as requested.")

    msgs_prompt: List[Dict[str, str]] = []
    msgs_full: List[Dict[str, str]] = []

    if system_prompt:
        msgs_prompt.append({"role": "system", "content": system_prompt})
        msgs_full.append({"role": "system", "content": system_prompt})

    msgs_prompt.append({"role": "user", "content": prompt})
    prompt_only_text = tokenizer.apply_chat_template(
        msgs_prompt,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
        enable_thinking=False,
    )

    msgs_full.append({"role": "user", "content": prompt})
    msgs_full.append({"role": "assistant", "content": completion})
    full_text = tokenizer.apply_chat_template(
        msgs_full,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
        enable_thinking=False,
    )
    return full_text, prompt_only_text

# ---------------------------
# Penalities
# ---------------------------

def rep_ngram_penalty(text: str, n: int = 4, level: str = "word") -> float:
    """
    Returns a penalty in [0, 1] roughly.
    0 = no repetition, 1 = extreme repetition.
    level: "word" or "char"
    """
    text = (text or "").strip()
    if not text:
        return 0.0

    if level == "word":
        toks = text.split()
    elif level == "char":
        toks = list(text)
    else:
        raise ValueError("level must be 'word' or 'char'")

    if len(toks) < n * 2:
        return 0.0

    ngrams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    if not ngrams:
        return 0.0

    c = Counter(ngrams)
    total = sum(c.values())
    unique = len(c)
    # repeated mass = 1 - unique/total
    penalty = 1.0 - (unique / total)
    return max(0.0, min(1.0, penalty))

def max_run_penalty(text: str, level: str = "char") -> float:
    """
    Penalize long runs of identical tokens (e.g., 0000000, hahahah, word word word).
    Returns penalty in [0, 1].
    """
    text = (text or "").strip()
    if not text:
        return 0.0

    toks = list(text) if level == "char" else text.split()
    if not toks:
        return 0.0

    max_run = 1
    run = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i-1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1

    # map run length to [0,1] with a soft curve
    # run 1..3 ~ small, run 20+ ~ near 1
    penalty = 1.0 - math.exp(-max(0, max_run - 3) / 8.0)
    return max(0.0, min(1.0, penalty))

_SENT_SPLIT = re.compile(r"(?:[.!?؟]+|\n+|[؛;]+|[。！？]+)\s*")

def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    # normalize spaces
    text = re.sub(r"\s+", " ", text)
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    return sents

def repeated_sentence_penalty(text: str, min_len_chars: int = 10) -> float:
    """
    Returns penalty in [0,1].
    0 = no repetition, 1 = all sentences are repeats.
    """
    sents = [s for s in split_sentences(text) if len(s) >= min_len_chars]
    if len(sents) <= 1:
        return 0.0

    # normalize lightly (spaces)
    norm = [re.sub(r"\s+", " ", s).strip() for s in sents]
    c = Counter(norm)

    total = len(norm)
    repeated_count = sum(v for v in c.values() if v >= 2)
    repeated_mass = repeated_count / total  # fraction of sentences that belong to a repeated group

    # also consider how concentrated the most frequent sentence is
    top_frac = max(c.values()) / total

    # combine: if one sentence dominates, it's bad
    penalty = max(repeated_mass, top_frac)
    return float(max(0.0, min(1.0, penalty)))


def looping_pattern_penalty(text: str, max_period: int = 6, min_repeats: int = 3) -> float:
    """
    Detects if the sentence sequence is largely a repeating pattern.
    Returns penalty in [0,1].
    """
    sents = split_sentences(text)
    if len(sents) < max(6, min_repeats * 2):
        return 0.0

    norm = [re.sub(r"\s+", " ", s).strip() for s in sents]
    n = len(norm)

    best_cover = 0.0
    for p in range(1, min(max_period, n // min_repeats) + 1):
        pattern = norm[:p]
        # check how much of the sequence matches repeating pattern
        matches = 0
        for i in range(n):
            if norm[i] == pattern[i % p]:
                matches += 1
        cover = matches / n
        # require that we actually repeat the pattern several times
        if n // p >= min_repeats:
            best_cover = max(best_cover, cover)

    # if 95% of sentences follow a short loop => strong penalty
    penalty = max(0.0, min(1.0, (best_cover - 0.6) / 0.4))  # maps 0.6..1.0 -> 0..1
    return penalty

def sentence_repetition_penalty(text: str) -> float:
    p_exact = repeated_sentence_penalty(text)
    p_loop  = looping_pattern_penalty(text)
    return max(p_exact, p_loop)


def word_repetition_penalty(text: str) -> float:
    p1 = rep_ngram_penalty(text, n=4, level="word")
    p2 = rep_ngram_penalty(text, n=8, level="char")
    p3 = max_run_penalty(text, level="char")
    return max(p1, p2, p3)

# ---------------------------
# Parallel file loading
# ---------------------------

JSONL_KEYS = ["completion", "output", "text", "hyp", "prediction"]


def _try_parse_json(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        return json.loads(line)
    except Exception:
        return None


def load_lines_keep_blanks(path: Path) -> List[str]:
    """
    Load a file as a list of lines, preserving line count.
    - For JSONL: each line should be a JSON object; we extract known keys.
    - For TXT: each line is the completion.
    Blank lines become "" (so alignment is preserved).
    """
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                out.append("")
                continue

            obj = _try_parse_json(line)
            if obj is not None:
                found = None
                for k in JSONL_KEYS:
                    if k in obj and isinstance(obj[k], str):
                        found = obj[k]
                        break
                out.append(found.strip() if found else "")
            else:
                out.append(line.strip())
    return out


def read_prompts_file(prompts_file: Path) -> List[str]:
    """
    One prompt per line. Blank lines are ignored by default (common in prompt lists).
    If you need to preserve blanks as actual prompts, remove the filter.
    """
    prompts: List[str] = []
    prompts = pd.read_csv(prompts_file, sep=',', encoding='utf-8')['prompt'].tolist()
    return prompts


def dedupe_keep_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        key = re.sub(r"\s+", " ", x.strip())
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


# ---------------------------
# Teacher forcing scoring
# ---------------------------
@dataclass
class TFScore:
    prompt_id: int
    prompt: str
    completion: str
    logp_sum: float
    token_count: int
    score: float
    source_file: str
    word_rep_penality: float
    sent_rep_penality: float


@torch.inference_mode()
def teacher_forced_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    system_prompt: Optional[str],
    device: str,
) -> Tuple[float, int]:
    full_text, prompt_only_text = build_full_and_prompt_only_text(
        tokenizer=tokenizer,
        prompt=prompt,
        completion=completion,
        system_prompt=system_prompt,
    )

    full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    prompt_ids = tokenizer(prompt_only_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    prompt_len = prompt_ids.shape[1]
    T = full_ids.shape[1]
    if prompt_len >= T:
        return 0.0, 0

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    outputs = model(input_ids=full_ids)
    logits = outputs.logits  # (1, T, V)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    if mask.sum().item() == 0:
        return 0.0, 0

    gathered = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.masked_fill(~mask, 0).unsqueeze(-1),
    ).squeeze(-1)

    logp_sum = gathered.masked_select(mask).sum().item()
    token_count = int(mask.sum().item())
    return logp_sum, token_count


def softmax_from_scores(scores: List[float], temperature: float) -> List[float]:
    if len(scores) == 0:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    Z = sum(exps)
    if Z == 0 or not math.isfinite(Z):
        return [1.0 / len(scores)] * len(scores)
    return [e / Z for e in exps]

# ---------------------------
# ChrF risk
# ---------------------------
def get_chrf_metric(word_order: int = 2, char_order: int = 6, beta: float = 2.0):
    try:
        from sacrebleu.metrics import CHRF  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not import sacrebleu. Install it with: pip install sacrebleu") from e
    return CHRF(word_order=word_order, char_order=char_order, beta=beta)


def chrf_0_1(metric, hyp: str, ref: str) -> float:
    return float(metric.sentence_score(hyp, [ref]).score) / 100.0


def compute_risks_chrf(
    completions: List[str],
    probs: List[float],
    *,
    metric,
) -> List[float]:
    n = len(completions)
    if n == 0:
        return []
    if len(probs) != n:
        raise ValueError("probs length mismatch")

    risks = [0.0 for _ in range(n)]
    for i in range(n):
        ri = 0.0
        hyp = completions[i]
        for j in range(n):
            pj = probs[j]
            ref = completions[j]
            sim = chrf_0_1(metric, hyp=hyp, ref=ref)
            ri += pj * (1.0 - sim)
        risks[i] = ri
    return risks


# ---------------------------
# Main CLI
# ---------------------------
@app.command()
def main(
    model: str = typer.Option(..., help="HF model name/path (CausalLM)."),
    prompts_file: Path = typer.Option(..., "--prompts-file", help="File with MANY prompts (one per line)."),
    outputs: List[Path] = typer.Option(..., "--outputs", "-o", help="Parallel output files (txt lines or jsonl)."),
    system_prompt: Optional[str] = typer.Option(None, help="Optional system prompt for chat template."),
    device: Optional[str] = typer.Option(None, help="cuda / cpu. Default: auto."),
    length_norm: bool = typer.Option(False, help="If set, use average logprob per token as score."),
    temperature: float = typer.Option(1.0, help="Softmax temperature for posterior p_model."),
    risk_temperature: float = typer.Option(1.0, help="Softmax temperature for p_risk = softmax(-risk / T)."),
    out: Path = typer.Option(Path("teacher_forcing_risk_parallel.csv"), help="Output CSV path."),
    out_text: Path = typer.Option(Path("out_text.csv"), help="Output CSV path for best."),
    out_json: Optional[Path] = typer.Option(None, help="Optional JSON output path."),
    strict_parallel: bool = typer.Option(
        False,
        help="If set, require each outputs file to have >= number of prompts lines (after prompt blank filtering).",
    ),
    dedupe_within_prompt: bool = typer.Option(
        False,
        help="If set, dedupe candidate completions within each prompt before scoring/risk.",
    ),
    chrf_word_order: int = typer.Option(2, help="ChrF word_order (sacrebleu)."),
    chrf_char_order: int = typer.Option(6, help="ChrF char_order (sacrebleu)."),
    chrf_beta: float = typer.Option(2.0, help="ChrF beta (sacrebleu)."),
):
    """
    Compute teacher-forced probabilities for each prompt's parallel candidates and ChrF-based MBR risk per prompt.
    """
    prompts = read_prompts_file(prompts_file)
    if len(prompts) == 0:
        raise RuntimeError("No prompts found in prompts-file (after removing blank lines).")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if not has_chat_template(tokenizer):
        raise RuntimeError("Tokenizer does not have a chat template; cannot proceed with apply_chat_template.")

    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float16 if device.startswith("cuda") else None,
        device_map="auto" if device.startswith("cuda") else None,
    )
    model_obj.eval()
    if not device.startswith("cuda"):
        model_obj.to(device)

    # Load all output files as aligned line arrays
    out_lines_by_file: Dict[str, List[str]] = {}
    for p in outputs:
        lines = load_lines_keep_blanks(p)
        out_lines_by_file[str(p)] = lines

    # Optionally enforce strict line coverage
    if strict_parallel:
        for fname, lines in out_lines_by_file.items():
            if len(lines) < len(prompts):
                raise RuntimeError(
                    f"File {fname} has only {len(lines)} lines but prompts-file has {len(prompts)} prompts."
                )

    chrf_metric = get_chrf_metric(word_order=chrf_word_order, char_order=chrf_char_order, beta=chrf_beta)

    # Collect all rows to write
    rows_csv: List[Dict[str, Any]] = []
    rows_json: List[Dict[str, Any]] = []
    best_candidates: List[str] = []

    for i, prompt_text in enumerate(prompts):
        # Gather candidates for prompt i from each outputs file
        candidates: List[Tuple[str, str]] = []  # (completion, source_file)
        for fname, lines in out_lines_by_file.items():
            comp = lines[i] if i < len(lines) else ""
            comp = (comp or "").strip()
            if comp:
                candidates.append((comp, fname))

        if not candidates:
            # No candidates for this prompt; skip (or you could write a placeholder row)
            continue

        if dedupe_within_prompt:
            # dedupe while keeping the first source occurrence
            seen = set()
            deduped: List[Tuple[str, str]] = []
            for comp, src in candidates:
                key = re.sub(r"\s+", " ", comp.strip())
                if key in seen:
                    continue
                seen.add(key)
                deduped.append((comp, src))
            candidates = deduped

        # Score each candidate under teacher forcing
        scored: List[TFScore] = []
        for comp, src in candidates:
            logp_sum, tok_n = teacher_forced_logprob(
                model=model_obj,
                tokenizer=tokenizer,
                prompt=prompt_text,
                completion=comp,
                system_prompt=system_prompt,
                device=device,
            )
            if length_norm:
                score = (logp_sum / tok_n) if tok_n > 0 else float("-inf")
            else:
                score = logp_sum

            scored.append(
                TFScore(
                    prompt_id=i,
                    prompt=prompt_text,
                    completion=comp,
                    logp_sum=logp_sum,
                    token_count=tok_n,
                    score=score,
                    source_file=src,
                    word_rep_penality=word_repetition_penalty(comp),
                    sent_rep_penality=sentence_repetition_penalty(comp)
                )
            )

        scores = [s.score for s in scored]
        p_model = softmax_from_scores(scores, temperature=temperature)

        # ChrF risks within this prompt's candidate set
        comps_only = [s.completion for s in scored]
        risks = compute_risks_chrf(comps_only, p_model, metric=chrf_metric)
        lambda_rep=0.1
        risk_final = [
            r + lambda_rep * s.word_rep_penality + lambda_rep * s.sent_rep_penality
            for r, s in zip(risks, scored)
        ]

        assert all(r >= 0 for r in risk_final)
        p_risk = softmax_from_scores(
            [-rf for rf in risk_final],
            temperature=risk_temperature,
        )

        # Rank by minimum risk (tie-break: higher p_model)
        order = sorted(
            range(len(scored)),
            key=lambda k: (-p_risk[k], -p_model[k])
        )
        best_candidates.append(scored[order[0]].completion)
        for rank, k in enumerate(order, start=1):
            s = scored[k]
            row = {
                "prompt_id": s.prompt_id,
                "rank_by_risk": rank,
                "prompt": s.prompt,
                "completion": s.completion,
                "token_count": s.token_count,
                "logp_sum": s.logp_sum,
                "score_used": s.score,
                "p_model": p_model[k],
                "risk_1_minus_chrf": risks[k],
                "p_risk": p_risk[k],
                "source": s.source_file,
                "sent_rep_penalty": s.sent_rep_penality,
                "word_rep_penality": s.word_rep_penality
            }
            rows_csv.append(row)
            rows_json.append(row)

    # Write CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "rank_by_risk",
        "prompt",
        "completion",
        "token_count",
        "logp_sum",
        "score_used",
        "p_model",
        "risk_1_minus_chrf",
        "p_risk",
        "source",
        "sent_rep_penalty",
        "word_rep_penality"
    ]

    if out_text is not None:
        with out_text.open("w", encoding="utf-8") as f:
            for line in best_candidates:
                f.write(line.replace("\n", "\\n") + "\n")

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f,
                           fieldnames=fieldnames,
                           quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows_csv:
            # format floats for readability
            r2 = dict(r)
            for k in ["logp_sum", "score_used", "p_model", "risk_1_minus_chrf", "p_risk"]:
                if isinstance(r2.get(k), float):
                    r2[k] = f"{r2[k]:.8f}"
            w.writerow(r2)


    # Optional JSON
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows_json, ensure_ascii=False, indent=2), encoding="utf-8")

    typer.echo(f"Wrote: {out}")
    if out_json is not None:
        typer.echo(f"Wrote: {out_json}")


if __name__ == "__main__":
    app()

