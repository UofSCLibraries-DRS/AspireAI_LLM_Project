import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch
from peft import LoraConfig, get_peft_model


def lora_ss(
    config_path: str,
    model_path: str,
    output_dir: str,
    data: pd.DataFrame,
):
    """
    Run LoRA (Low-Rank Adaptation) Self-Supervised training using the given configuration.

    Args:
        config_path (str): Path to the training configuration file (JSON).
        model_path (str): Absolute path to the base model weights that will be fine-tuned.
        output_dir (str): Directory where all training outputs will be written.
            The following subdirectories are created automatically:
                - ``scratch/`` : Intermediate checkpoints during training.
                - ``logs/`` : Training logs and history (``log_history.json``).
                - ``model/`` : Final trained model weights and tokenizer.
        data (pd.DataFrame): Pandas DataFrame containing the training corpus.
            Must include a column named ``"text"`` with raw text samples.

    Returns:
        None
    """
    # Load cfg
    with open(config_path, "r") as f:
        cfg = json.load(f)

    lora_cfg = cfg["lora_config"]
    training_cfg = cfg["training_args"]

    dataset = Dataset.from_pandas(data)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=1024
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Load local model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Apply Lora cfg
    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, lora_config)

    # Add training args
    training_args = TrainingArguments(
        **training_cfg,
        output_dir=f"{output_dir}/scratch",
        logging_dir=f"{output_dir}/logs",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Traing and save
    trainer.train()

    trainer.save_model(f"{output_dir}/model")
    tokenizer.save_pretrained(f"{output_dir}/model")

    with open(f"{output_dir}/logs/log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f)
