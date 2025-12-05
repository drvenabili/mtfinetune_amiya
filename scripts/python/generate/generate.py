import typer
from enum import Enum
from typing import Optional

import torch
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
            model_path
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
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

    # Build chat messages – this assumes you trained with chat format via setup_chat_format
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Use the tokenizer’s chat template (set during training with setup_chat_format)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
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

if __name__ == "__main__":
    app()

