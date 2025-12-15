import typer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

app = typer.Typer()

@app.command()
def testupload(
    model_name: str = typer.Option("HuggingFaceTB/SmolLM3-3B", help="The name of the model to download."),
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

if __name__ == "__main__":
    app()
