import csv
import yaml
import json
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import os
import shutil
from peft import LoraConfig, get_peft_model

from .training_base import AbstractTrainer
from fine_tuning.utils.environment import get_env_or_raise

MAX_LENGTH = 512


def build_prompt(question: str, answer: str, prompt_cfg: dict) -> str:
    """Prompt template for supervised fine-tuning."""
    return (
        prompt_cfg["template"].format(user_prompt=question)
        + answer
        + prompt_cfg["template_stop"]
    )


def prepare_example(example, tokenizer, max_length, prompt_cfg):
    """Tokenize and mask question part (labels = -100 for prompt tokens)."""
    prompt = build_prompt(example["question"], example["answer"], prompt_cfg)
    answer = example["answer"]
    full_text = prompt + answer

    # Tokenize
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = [-100] * len(input_ids)

    # Only compute loss on answer tokens
    non_padded_len = sum(attention_mask)
    start_idx = min(len(prompt_ids), non_padded_len)
    for i in range(start_idx, non_padded_len):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class LoRASFTTrainer(AbstractTrainer):
    def _train(self):
        with open(self.config, "r") as f:
            cfg = json.load(f)

        lora_cfg = cfg["lora_config"]
        training_cfg = cfg["training_args"]
        prompt_cfg_path = os.path.join(
            get_env_or_raise("PROMPT_FOLDER"), cfg["prompt_format"]
        )

        with open(prompt_cfg_path, "r") as f:
            prompt_cfg = yaml.safe_load(f)

        with open(self.data, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                {"question": r["question"], "answer": r["answer"]}
                for r in reader
                if r.get("question")
                and r.get("answer")  # Drop rows with missing values
            ]

        # Create Hugging Face dataset
        dataset = Dataset.from_dict(
            {
                "question": [r["question"] for r in rows],
                "answer": [r["answer"] for r in rows],
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(self.start_model, use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_batch(examples):
            out = {"input_ids": [], "attention_mask": [], "labels": []}
            for q, a in zip(examples["question"], examples["answer"]):
                ex = {"question": q, "answer": a}
                prepared = prepare_example(ex, tokenizer, MAX_LENGTH, prompt_cfg)
                for k in out:
                    out[k].append(prepared[k])
            return out

        tokenized_dataset = dataset.map(
            tokenize_batch, batched=True, remove_columns=["question", "answer"]
        )

        # Load local model
        model = AutoModelForCausalLM.from_pretrained(
            self.start_model,
            torch_dtype="auto",
            device_map="cuda",
        )

        # Apply Lora cfg
        lora_config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            **training_cfg,
            output_dir=f"{self.output_dir}/scratch",
            logging_dir=f"{self.output_dir}/logs",
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train and save
        trainer.train()

        # Save LoRA adapters
        adapter_dir = os.path.join(self.output_dir, "adapters")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(
            self.output_dir
        )  # Double saving the tokenizer for convenience

        # Save the full model with adapters applied (Useful when applying multiple stages of LoRA)
        model = model.merge_and_unload()
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(
            self.output_dir
        )  # Double saving the tokenizer for convenience

        # Save prompt format
        shutil.copy2(
            prompt_cfg_path,
            os.path.join(self.output_dir, "prompt.yaml"),
        )

        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)

        with open(f"{self.output_dir}/logs/log_history.json", "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)

        loss_values = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        steps = [x["step"] for x in trainer.state.log_history if "loss" in x]

        if len(loss_values) > 1:
            plt.figure(figsize=(8, 5))
            plt.plot(steps, loss_values, label="Training Loss", linewidth=2)
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve (Gemma 270M Full Fine-tuning)")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            plot_path = f"{self.output_dir}/logs/loss_plot.png"
            plt.savefig(plot_path)
        else:
            print("No loss data found in trainer.state.log_history — skipping plot.")
