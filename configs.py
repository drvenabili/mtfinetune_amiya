from peft import LoraConfig
from trl import SFTConfig

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
