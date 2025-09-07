import argparse
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


def main():
    # Handle args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to config JSON file"
    )
    args = parser.parse_args()

    # Load cfg
    with open(args.cfg, "r") as f:
        cfg = json.load(f)

    lora_cfg = cfg["lora_config"]
    training_cfg = cfg["training_args"]
    local_model_path = cfg["local_model_path"]
    final_output_dir = cfg["final_output_dir"]

    # Load data
    df = pd.read_csv("/work/jaaydin/100_manually_cleaned.csv")
    df = df[["Cleaned Transcript"]].dropna()
    dataset = Dataset.from_pandas(df.rename(columns={"Cleaned Transcript": "text"}))

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=1024
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Load local model
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, torch_dtype=torch.float16, device_map="auto"
    )

    # Apply Lora cfg
    lora_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, lora_config)

    # Add training args
    training_args = TrainingArguments(**training_cfg)

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

    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    main()
