import csv
import json
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import os
import torch

from .training_base import AbstractTrainer


class FullUnsupervisedTrainer(AbstractTrainer):
    def _train(self):
        with open(self.config, "r") as f:
            cfg = json.load(f)

        training_cfg = cfg["training_args"]

        with open(self.data, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames

            # Check for the "text" column
            if "text" not in columns:
                raise ValueError(
                    f"CSV must contain a 'text' column. Columns found: {columns}"
                )

            # Collect values
            text_values = []
            for row in reader:
                val = row.get("text")
                if val is not None:
                    text_values.append(str(val))

        # Convert to a Hugging Face dataset
        dataset = Dataset.from_dict({"text": text_values})

        tokenizer = AutoTokenizer.from_pretrained(self.start_model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=1024,
            )

        tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

        # Load local model
        model = AutoModelForCausalLM.from_pretrained(
            self.start_model,
            torch_dtype="auto",
            device_map="cuda",
            attn_implementation="eager",
        )

        for param in model.parameters():
            param.requires_grad = True

        model.gradient_checkpointing_enable()

        training_args = TrainingArguments(
            **training_cfg,
            output_dir=f"{self.output_dir}/scratch",
            logging_dir=f"{self.output_dir}/logs",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Traing and save
        trainer.train()

        trainer.save_model(f"{self.output_dir}")
        tokenizer.save_pretrained(f"{self.output_dir}")

        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)

        with open(f"{self.output_dir}/logs/log_history.json", "w") as f:
            json.dump(trainer.state.log_history, f)

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
            print(
                "WARNING: No loss data found in trainer.state.log_history — skipping plot."
            )
