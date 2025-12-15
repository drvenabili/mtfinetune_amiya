from peft import LoraConfig
from trl import SFTConfig
from typing import Callable, Dict

def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

def get_sft_config(method: str):
    return SFTConfig(
        output_dir=f"./results_{method}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=2048,
    )

def smollm_translation_format(
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str = "",
) -> str:
    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}.\n\n"
        f"{source_lang}:\n{source_text}\n\n"
        f"{target_lang}:\n"
    )
    if target_text:
        return prompt + target_text
    return prompt


def llama_instruct_format(
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str = "",
) -> str:
    prompt = (
        "<|begin_of_text|>\n"
        "<|user|>\n"
        f"Translate the following text from {source_lang} to {target_lang}.\n\n"
        f"{source_lang}:\n{source_text}\n"
        "<|assistant|>\n"
    )
    if target_text:
        return prompt + target_text
    return prompt


def get_model_config(model_name: str = "HuggingFaceTB/SmolLM3-3B") -> Dict:
    if model_name == "HuggingFaceTB/SmolLM3-3B":
        chat_format = smollm_translation_format

    elif model_name in {
        "meta-llama/Llama-3-8B-Instruct",
        "meta-llama/Llama-3-70B-Instruct",
    }:
        chat_format = llama_instruct_format

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return {
        "model_name": model_name,
        "chat_format": chat_format,
    }

