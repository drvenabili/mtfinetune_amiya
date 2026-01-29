#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import os
from datasets import Dataset
from transformers import AutoTokenizer

app = typer.Typer(add_completion=False, help="Prepare a prompt/completion CSV for instruct fine-tuning.")

DEFAULT_CHAT_TEMPLATE = r"""
{% for message in messages %}
{% if message['role'] == 'system' -%}
<|system|>
{{ message['content'] | trim }}
{% elif message['role'] == 'user' -%}
<|user|>
{{ message['content'] | trim }}
{% elif message['role'] == 'assistant' -%}
<|assistant|>
{{ message['content'] | trim }}
{% endif -%}
{{ '\n' }}
{% endfor %}
{% if add_generation_prompt -%}
<|assistant|>
{% endif -%}
""".strip()

def debug_chat_template(messages, tokenizer):
    """Debug chat template application"""
    # Apply template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and decode to see actual tokens
    tokens = tokenizer(formatted, return_tensors="pt")

    print("=== TEMPLATE DEBUG ===")
    print(f"Input messages: {len(messages)}")
    print(f"Formatted length: {len(formatted)} chars")
    print(f"Token count: {tokens['input_ids'].shape[1]}")
    print("\nFormatted text:")
    print(repr(formatted))  # Shows escape characters
    print("\nTokens:")
    print(tokens['input_ids'][0].tolist()[:20], "...")  # First 20 tokens
    print("\nDecoded tokens:")
    for i, token_id in enumerate(tokens['input_ids'][0][:20]):
        token = tokenizer.decode([token_id])
        print(f"{i:2d}: {token_id:5d} -> {repr(token)}")

# Example usage
#debug_messages = [
#    {"role": "user", "content": "Hello!"},
#    {"role": "assistant", "content": "Hi there!"}
#]

#debug_chat_template(debug_messages, tokenizer)


def normalize_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x)
    # handle common "nan" string when reading CSVs
    if s.lower() == "nan":
        return ""
    return s.strip()

def has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))

def set_or_override_chat_template(
    tokenizer,
    template_path: Optional[Path],
    override: bool,
    add_special_tokens: bool,
) -> None:
    if has_chat_template(tokenizer) and not override:
        return

    if template_path:
        template = template_path.read_text(encoding="utf-8").strip()
    else:
        template = DEFAULT_CHAT_TEMPLATE

    tokenizer.chat_template = template

    if add_special_tokens:
        specials = {"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]}
        tokenizer.add_special_tokens(specials)

def build_text_pair(
    tokenizer,
    prompt: str,
    completion: str,
    system_prompt: Optional[str]
) -> Optional[tuple[str, str]]:
    prompt = normalize_text(prompt)
    completion = normalize_text(completion)
    if not prompt or not completion:
        return None

    if has_chat_template(tokenizer):
        msgs_prompt = []
        msgs_full = []
        if system_prompt:
            msgs_prompt.append({"role": "system", "content": system_prompt})
            msgs_full.append({"role": "system", "content": system_prompt})

        msgs_prompt.append({"role": "user", "content": prompt})
        prompt_only_text = tokenizer.apply_chat_template(
            msgs_prompt,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            enable_thinking=False
        )

        msgs_full.append({"role": "user", "content": prompt})
        msgs_full.append({"role": "assistant", "content": completion})
        full_text = tokenizer.apply_chat_template(
            msgs_full,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
            enable_thinking=False
        )
        return full_text, prompt_only_text

    # Non-chat fallback (won't happen if we set template, but kept for safety)
    print("not template can be added")
    print(f"### Instruction:\n{prompt}\n\n### Response:\n")
    prompt_only_text = f"### Instruction:\n{prompt}\n\n### Response:\n"
    full_text = prompt_only_text + completion
    return full_text, prompt_only_text

def preprocess_row(
    example: dict,
    tokenizer,
    max_length: int,
    system_prompt: Optional[str]
) -> dict:
    built = build_text_pair(tokenizer, example["prompt"], example["completion"], system_prompt)
    if built is None:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    full_text, prompt_only_text = built

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    prompt_only = tokenizer(
        prompt_only_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    # we want to ignore the instruction prompt in the loss function
    prompt_len = len(prompt_only["input_ids"])
    cut = min(prompt_len, len(input_ids))
    labels = [-100] * cut + input_ids[cut:]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@app.command("tokenize")
def tokenize_cmd(
    csv_path: Path = typer.Argument(..., exists=True, readable=True, help="CSV with columns: prompt, completion"),
    model_id: str = typer.Option(..., help="Hugging Face model id (used to load tokenizer or path)."),
    out_dir: Path = typer.Option(Path("tokenized_dataset"), help="Output directory (HF datasets format)."),
    max_length: int = typer.Option(2048, help="Max sequence length."),
    system_prompt: Optional[str] = typer.Option("You are a helpful assistant.", help="System prompt (or empty to disable)."),
    delimiter: str = typer.Option(",", help="CSV delimiter (use '\\t' for TSV)."),
    encoding: str = typer.Option("utf-8", help="CSV encoding."),
    text_col_prompt: str = typer.Option("prompt", help="Prompt column name."),
    text_col_completion: str = typer.Option("completion", help="Completion column name."),
    create_template_if_missing: bool = typer.Option(True, help="If tokenizer has no chat_template, set a default one."),
    template_path: Optional[Path] = typer.Option(None, help="Path to a Jinja chat template to use."),
    override_existing_template: bool = typer.Option(False, help="Override tokenizer.chat_template even if it exists."),
    add_chat_special_tokens: bool = typer.Option(False, help="Add <|system|>/<|user|>/<|assistant|> as special tokens."),
    save_tokenizer_with_template: Optional[Path] = typer.Option(
        None, help="If set, also save the tokenizer (with chat_template) to this directory."
    ),
    preview_n: int = typer.Option(1, help="Print a preview of formatted text for the first N rows."),
    save_config_json: bool = typer.Option(True, help="Save preprocessing config next to the dataset."),
    num_proc: Optional[int] = typer.Option(None, help="Number of processes for dataset.map (defaults to SLURM_CPUS_PER_TASK or CPU count)."),
    batch_size: Optional[int] = typer.Option(128, help="Numbero of batch size to preprocess")
):
    # Load data
    df = pd.read_csv(csv_path, sep=delimiter, encoding=encoding, engine="python")

    if text_col_prompt not in df.columns or text_col_completion not in df.columns:
        raise typer.BadParameter(
            f"Missing required columns. Found: {list(df.columns)}; "
            f"need: {text_col_prompt}, {text_col_completion}"
        )

    df = df.rename(columns={text_col_prompt: "prompt", text_col_completion: "completion"})
    df["prompt"] = df["prompt"].map(normalize_text)
    df["completion"] = df["completion"].map(normalize_text)
    df = df[(df["prompt"] != "") & (df["completion"] != "")].reset_index(drop=True)

    if len(df) == 0:
        raise typer.BadParameter("No non-empty rows after cleaning prompt/completion.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if create_template_if_missing:
        set_or_override_chat_template(
            tokenizer=tokenizer,
            template_path=template_path,
            override=override_existing_template,
            add_special_tokens=add_chat_special_tokens,
        )

    # Preview formatting
    if preview_n > 0:
        n = min(preview_n, len(df))
        typer.echo(f"\n--- Preview ({n} rows) ---")
        for i in range(n):
            full_text, prompt_only_text = build_text_pair(
                tokenizer, df.loc[i, "prompt"], df.loc[i, "completion"], system_prompt if system_prompt else None
            )
            typer.echo(f"\n[Row {i}] PROMPT_ONLY:\n{prompt_only_text}")
            typer.echo(f"\n[Row {i}] FULL:\n{full_text}")
        typer.echo("\n--- End preview ---\n")

    ds = Dataset.from_pandas(df)

    tokenized = ds.map(
        lambda ex: preprocess_row(ex, tokenizer, max_length=max_length, system_prompt=system_prompt if system_prompt else None),
        remove_columns=ds.column_names,
        num_proc=num_proc,
        batch_size=128,
        desc="Tokenizing",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(out_dir))
    typer.echo(f"Saved tokenized dataset to: {out_dir}")

    # Save tokenizer if requested (keeps chat_template for future runs)
    if save_tokenizer_with_template is not None:
        save_tokenizer_with_template.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(save_tokenizer_with_template))
        typer.echo(f"Saved tokenizer (with chat_template) to: {save_tokenizer_with_template}")

    # Save config JSON (for reproducibility)
    if save_config_json:
        cfg = {
            "csv_path": str(csv_path),
            "model_id": model_id,
            "out_dir": str(out_dir),
            "max_length": max_length,
            "system_prompt": system_prompt,
            "delimiter": delimiter,
            "encoding": encoding,
            "create_template_if_missing": create_template_if_missing,
            "template_path": str(template_path) if template_path else None,
            "override_existing_template": override_existing_template,
            "add_chat_special_tokens": add_chat_special_tokens,
            "tokenizer_pad_token": tokenizer.pad_token,
            "tokenizer_eos_token": tokenizer.eos_token,
            "has_chat_template": has_chat_template(tokenizer),
        }
        (out_dir / "preprocess_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"Wrote config: {out_dir / 'preprocess_config.json'}")

if __name__ == "__main__":
    app()

