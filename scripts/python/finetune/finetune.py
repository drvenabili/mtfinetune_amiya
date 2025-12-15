import typer
from enum import Enum
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format
from configs import get_lora_config, get_sft_config, get_model_config
from huggingface_hub import login

app = typer.Typer()

class TuningMethod(str, Enum):
    trl = "trl"
    lora = "lora"

@app.command()
def finetune(
    method: TuningMethod = typer.Option(TuningMethod.trl, help="The finetuning method to use: 'trl' (full finetuning) or 'lora'."),
    model_name: str = typer.Option("HuggingFaceTB/SmolLM3-3B", help="The name of the model to finetune."),
    dataset_name: str = typer.Option("fillwith/realdata", help="The name of the dataset to use for finetuning."),
):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device} with method {method.value} for model {model_name}")

    # Get model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token is set (required for SFTTrainer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = get_model_config(model_name)

    # Define formatting function
    def format_example(example):
        # Fix this for format of dataset
        src_lang = example.get("source_language") or example.get("source_lang") or "English"
        tgt_lang = example.get("target_language") or example.get("target_lang") or "French"
        src_text = example.get("source_text") or example.get("source") or example.get("prompt") 
        tgt_text = example.get("target_text") or example.get("target") or example.get("completion")
        
        if src_text is None:
             return {"text": ""}

        text = cfg["chat_format"](
            source_lang=src_lang,
            target_lang=tgt_lang,
            source_text=src_text,
            target_text=tgt_text
        )
        return {"text": text}

    # get data
    dataset = load_dataset(dataset_name)
    
    # Handle splits if necessary (basic check)
    if "validation" not in dataset and "test" in dataset:
        dataset["validation"] = dataset["test"]
    elif "validation" not in dataset:
        # Simple split if no validation set exists
        dataset = dataset.train_test_split(test_size=0.1)
        dataset["validation"] = dataset.pop("test")

    dataset = dataset.map(
        format_example,
        remove_columns=dataset["train"].column_names,
    )

    # Configure LoRA if selected
    peft_config = None
    if method == TuningMethod.lora:
        peft_config = get_lora_config()

    # sfttrainer
    sft_config = get_sft_config(method.value)
    sft_config.dataset_text_field = "text"
    
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