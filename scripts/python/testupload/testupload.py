import typer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments
from huggingface_hub import login, ModelCard

import json
import torch
from transformers import 
from huggingface_hub import 

app = typer.Typer()

def extract_metrics(trainer_state):
    metrics = {}
    if "best_metric" in trainer_state:
        metrics["best_metric"] = trainer_state["best_metric"]
    if "log_history" in trainer_state:
        final_logs = [x for x in trainer_state["log_history"] if "eval_loss" in x]
        if final_logs:
            metrics.update(final_logs[-1])
    return metrics

@app.command()
def testupload(
    model_name: str = typer.Option("HuggingFaceTB/SmolLM3-3B", help="The name of the model to upload."),
    org_name: str = typer.Option("unige-fti", help="The organization to upload to."),
    private: bool = typer.Option(True, help="Whether the uploaded model should be private.")
):
    print(f"Downloading {model_name}...")
    
    # Login
    if os.path.exists("hf_token"):
        with open("hf_token", "r") as f:
            token = f.read().strip()
            login(token)
            print("Logged in to Hugging Face Hub.")
    else:
        print("Warning: hf_token file not found. Assuming already logged in or public model.")

    # Download model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    repo_name = model_name.split("/")[-1]
    target_repo = f"{org_name}/{repo_name}"
    
    print(f"Uploading to {target_repo} (private={private})...")
    
    # Upload
    model.push_to_hub(target_repo, private=private)
    tokenizer.push_to_hub(target_repo, private=private)
    
    print("Done!")

@app.command()
def generate_model_card(MODEL_DIR: str = typer.Option("./model", help="Path to the trained model directory"), 
                        OUTPUT_README: str = typer.Option("./model/README.md", help="Path to save the generated README.md")):
    
    config = AutoConfig.from_pretrained(MODEL_DIR)
    training_args = TrainingArguments.load(os.path.join(MODEL_DIR, "training_args.bin"))
    with open(os.path.join(MODEL_DIR, "trainer_state.json"), "r") as f:
        trainer_state = json.load(f)

    metrics = extract_metrics(trainer_state)

    tokenizer_info = None
    tok_cfg_path = os.path.join(MODEL_DIR, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path) as f:
            tokenizer_info = json.load(f)


    card_data = {
        "language": "en",
        "license": "apache-2.0",  # CHANGE IF NEEDED
        "library_name": "transformers",
        "base_model": config._name_or_path,
        "tags": [
            "fine-tuned",
            config.model_type,
        ],
        "model_name": os.path.basename(os.path.abspath(MODEL_DIR)),
    }
    training_table = f"""
| Parameter | Value |
|----------|-------|
| Epochs | {training_args.num_train_epochs} |
| Train batch size | {training_args.per_device_train_batch_size} |
| Eval batch size | {training_args.per_device_eval_batch_size} |
| Learning rate | {training_args.learning_rate} |
| Optimizer | {training_args.optim} |
| FP16 | {training_args.fp16} |
| BF16 | {training_args.bf16} |
"""

    metrics_md = "\n".join(
        f"- **{k}**: {v}" for k, v in metrics.items()
    ) if metrics else "No evaluation metrics available."

    tokenizer_md = (
        f"Tokenizer class: `{tokenizer_info.get('tokenizer_class')}`"
        if tokenizer_info else
        "Tokenizer information not available."
    )

    body = f"""
## Model description

This model is a fine-tuned version of **{config._name_or_path}** using the 🤗 Transformers `Trainer` API.

## Model architecture

- Model type: `{config.model_type}`
- Hidden size: `{getattr(config, 'hidden_size', 'N/A')}`
- Number of layers: `{getattr(config, 'num_hidden_layers', 'N/A')}`
- Number of attention heads: `{getattr(config, 'num_attention_heads', 'N/A')}`

## Training procedure

### Hyperparameters
{training_table}

### Training results
{metrics_md}

## Tokenizer
{tokenizer_md}

## Intended use

This model is intended for research and experimentation.  
Please evaluate carefully before using in production.

## Limitations and bias

The model may reflect biases present in the training data and the base model.

## Citation

If you use this model, please cite the original base model and the Transformers library.
"""

    # -------- Create and save model card --------
    card = ModelCard.from_template(
        card_data=card_data,
        template_str=body,
    )

    card.save(OUTPUT_README)
    print(f"✅ Model card written to {OUTPUT_README}")


if __name__ == "__main__":
    app()