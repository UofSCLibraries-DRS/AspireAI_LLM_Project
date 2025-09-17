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

    output_dir = cfg["output_dir"]

    lora_cfg = cfg["lora_config"]
    training_cfg = cfg["training_args"]
    local_model_path = cfg["local_model_path"]

    # Load data
    df = pd.read_csv("/work/jaaydin/data/1940s_mccray.csv")
    df = df[["Original Transcript"]].dropna()
    dataset = Dataset.from_pandas(df.rename(columns={"Original Transcript": "text"}))

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


if __name__ == "__main__":
    main()
