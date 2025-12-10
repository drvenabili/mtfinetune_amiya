import typer
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from configs import get_lora_config, get_sft_config
from huggingface_hub import login
import os

app = typer.Typer()

class TuningMethod(str, Enum):
    trl = "trl"
    lora = "lora"

@app.command()
def main(
    method: TuningMethod = typer.Option(TuningMethod.trl, help="The finetuning method to use: 'trl' (full finetuning) or 'lora'."),
    model_name: str = typer.Option("HuggingFaceTB/SmolLM3-3B", help="The name of the model to finetune.")
):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} with method {method.value} for model {model_name}")

    # Get model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # setup chat format
    model, tokenizer = setup_chat_format(model, tokenizer)


    # get data
    dataset = load_dataset("fillwith/realdata")

    # Configure LoRA if selected
    peft_config = None
    if method == TuningMethod.lora:
        peft_config = get_lora_config()

    # sfttrainer
    sft_config = get_sft_config(method.value)
    trainer = SFTTrainer(
        model=model,
        args = sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # main training loop
    trainer.train()
    trainer.save_model(f"{model_name.split('/')[-1]}-aladdinFTI-sft-{method.value}")
    try:
        trainer.push_to_hub(f"unige-fti/{model_name.split('/')[-1]}-aladdinFTI-sft-{method.value}", private=True)
    except Exception as e:
        print(f"Could not push to hub: {e}")

if __name__ == "__main__":
    print(f"Trying to log in to Hugging Face Hub...")
    if not os.path.exists("hf_token"):
        print("""##############
No hf_token file found. 
Resulting model will not be uploaded to the hub.
##############""")
    else:
        with open("hf_token", "r") as f:
            token = f.read().strip()
            login(token)
            print("Logged in to Hugging Face Hub.")

    app()