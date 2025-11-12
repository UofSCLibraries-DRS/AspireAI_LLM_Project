import csv
import json
import os
import gc
from accelerate import Accelerator
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

from .training_base import AbstractTrainer


class LoRAUnsupervisedTrainer(AbstractTrainer):
    def _train(self):
        with open(self.config, "r") as f:
            cfg = json.load(f)

        lora_cfg = cfg["lora_config"]
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
            return tokenizer(  # noqa: F821
                batch["text"], truncation=True, padding="max_length", max_length=1024
            )

        tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

        # Load local model
        model = AutoModelForCausalLM.from_pretrained(
            self.start_model, torch_dtype="auto", device_map="auto"
        )

        # Apply Lora cfg
        lora_config = LoraConfig(**lora_cfg)
        model = get_peft_model(model, lora_config)

        # Add training args
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

        # Traing and save
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

        os.makedirs(f"{self.output_dir}/logs", exist_ok=True)

        with open(f"{self.output_dir}/logs/log_history.json", "w") as f:
            json.dump(trainer.state.log_history, f)

        del model, trainer, tokenizer, dataset, tokenized_dataset

        gc.collect()
        torch.cuda.empty_cache()
        Accelerator().free_memory()
