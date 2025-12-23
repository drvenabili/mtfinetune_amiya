import typer
from enum import Enum
from typing import Optional, List
from pathlib import Path

from collections.abc import Mapping
import torch

import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM  # needed for LoRA models

app = typer.Typer()

class TuningMethod(str, Enum):
    trl = "trl"   # full finetuning
    lora = "lora" # LoRA / PEFT
    base = "base" # no fine-tuning

def _load_model_and_tokenizer(
    model_path: str,
    method: TuningMethod,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # For LoRA, load with AutoPeftModelForCausalLM
    if method == TuningMethod.lora:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # if we are generating, better to be in the left
    tokenizer.padding_side = "left"
    if "Llama" in model_path:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    print(f"model to device {device}")
    model.eval()
    return model, tokenizer, device

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="User prompt to send to the chat model."),
    model_path: str = typer.Option(
        "SmolLM3-3B-aladdinFTI-sft-trl",
        help=(
            "Path or Hub name of the fine-tuned model, e.g. "
            "'SmolLM3-3B-aladdinFTI-sft-trl' or 'SmolLM3-3B-aladdinFTI-sft-lora'."
        ),
    ),
    method: TuningMethod = typer.Option(
        TuningMethod.trl,
        help="Tuning method used for this model: 'trl' (full finetuning), 'lora'"
        "or if not 'base'",
    ),
    max_new_tokens: int = typer.Option(256, help="Maximum number of new tokens to generate."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling cut-off."),
    seed: Optional[int] = typer.Option(111, help="Random seed for reproducibility."),
):
    """
    Generate a response using a fine-tuned LLM.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model, tokenizer, device = _load_model_and_tokenizer(model_path, method)
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    else:
        input_ids = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens
    generated_ids = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\n=== MODEL RESPONSE ===\n")
    print(text)
    print("\n======================\n")

@app.command("generate-batch")
def generate_batch(
    prompts_file: Path = typer.Argument(
        ...,
        help="Text file with one prompt per line.",
    ),
    model_path: str = typer.Option(
        "SmolLM3-3B-aladdinFTI-sft-trl",
        help=(
            "Path or Hub name of the fine-tuned model, e.g. "
            "'SmolLM3-3B-aladdinFTI-sft-trl' or 'SmolLM3-3B-aladdinFTI-sft-lora'."
        ),
    ),
    method: TuningMethod = typer.Option(
        TuningMethod.trl,
        help="Tuning method used for this model: 'trl' (full finetuning), 'lora' or 'base'.",
    ),
    batch_size: int = typer.Option(
        8, help="Number of prompts to process in parallel."
    ),
    max_new_tokens: int = typer.Option(512, help="Maximum number of new tokens to generate."),
    temperature: float = typer.Option(0.7, help="Sampling temperature."),
    top_p: float = typer.Option(0.9, help="Top-p nucleus sampling cut-off."),
    seed: Optional[int] = typer.Option(111, help="Random seed for reproducibility."),
    reasoning_mode: Optional[bool] = typer.Option(False, help="Reasoning mode (thinking mode)"),
    output_file: Optional[Path] = typer.Option(
        None,
        help="Optional path to save outputs as TSV: prompt<TAB>response.",
    ),
):
    """
    Generate responses for many prompts using batched inference.
    """
    # Read prompts
    do_sample: bool = True
    if temperature == 0:
        do_sample = False
    prompts: List[str] = []
    df = pd.read_csv(prompts_file, sep=',')
    prompts = df['prompt'].tolist()
    if not prompts:
        typer.echo("No prompts found in the file.")
        raise typer.Exit(code=1)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model, tokenizer, device = _load_model_and_tokenizer(model_path, method)

    all_responses: List[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]

        if getattr(tokenizer, "chat_template", None):
            # Build chat messages per prompt
            print("Batch template!")
            batch_messages = [
                [{"role": "user", "content": p}] for p in batch_prompts
            ]
            # Tokenize with padding so we can batch
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                enable_thinking=reasoning_mode,
            )
        else:
            print("No batch template")
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt"
            ).to(device)
        # Move to device (inputs is usually a dict with input_ids, attention_mask)
        if isinstance(inputs, Mapping):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                print("attention_mask is none")
                ttention_mask = torch.ones_like(input_ids, device=device)
        else:
            # Fallback if tokenizer returns a tensor directly
            input_ids = inputs.to(device)
            # Build a full-ones attention mask (no padding case)
            attention_mask = torch.ones_like(input_ids, device=device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        # For each item in the batch, cut off the input part.
        # We use attention_mask to get the real input length per example.
        for i, prompt_text in enumerate(batch_prompts):
            input_len = int(attention_mask[i].sum().item())
            generated_ids = output_ids[i, input_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_responses.append(text)

            #print("\n=== EXAMPLE", start + i, "===\n")
            #print("PROMPT:")
            #print(prompt_text)
            #print("\nRESPONSE:")
            #print(text)
            #print("\n======================\n")

    # Optionally save to file
    if output_file is not None:
        with output_file.open("w", encoding="utf-8") as f:
            for r in all_responses:
                # simple TSV: prompt<TAB>response
                f.write(r.replace("\n", "\\n") + "\n")
        typer.echo(f"Saved {len(all_responses)} examples to {output_file}")

if __name__ == "__main__":
    app()

