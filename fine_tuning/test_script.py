import csv
import json
import os
import gc
from accelerate import Accelerator
from datasets import Dataset
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import torch
from peft import LoraConfig, get_peft_model


# Set up folders
OUTPUT_DIR = "/work/jaaydin/models/test_gemma"
CONFIG = "/work/jaaydin/AspireAI_LLM_Project/fine_tuning/config/ft/training/lora_unsupervised_D_mcc_h2.json"
DATA = "/work/jaaydin/raw/100_transcript.csv"
START_MODEL = "/work/jaaydin/models/gemma-3-270m"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
SCRATCH = os.path.join(OUTPUT_DIR, "scratch")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "adapters")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)

print("1")

with open(CONFIG, "r") as f:
    cfg = json.load(f)

lora_cfg = cfg["lora_config"]
training_cfg = cfg["training_args"]

print("2")

with open(DATA, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    columns = reader.fieldnames

    # Check for the "text" column
    if "text" not in columns:
        raise ValueError(f"CSV must contain a 'text' column. Columns found: {columns}")

    # Collect values
    text_values = []
    for row in reader:
        val = row.get("text")
        if val is not None:
            text_values.append(str(val))

print("3")

# Convert to a Hugging Face dataset
dataset = Dataset.from_dict({"text": text_values})

print("4")

tokenizer = AutoTokenizer.from_pretrained(START_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    return tokenizer(  # noqa: F821
        batch["text"], truncation=True, padding="max_length", max_length=1024
    )


tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Load local model
model = AutoModelForCausalLM.from_pretrained(
    START_MODEL, torch_dtype="auto", device_map="auto"
)

print(next(model.parameters()).device)

# Apply Lora cfg
lora_config = LoraConfig(**lora_cfg)
model = get_peft_model(model, lora_config)

# Add training args
training_args = TrainingArguments(
    **training_cfg,
    output_dir=SCRATCH,
    logging_dir=LOG_DIR,
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
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)  # Double saving the tokenizer for convenience

# Save the full model with adapters applied (Useful when applying multiple stages of LoRA)
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)  # Double saving the tokenizer for convenience

with open(os.path.join(LOG_DIR, "log_history.json"), "w") as f:
    json.dump(trainer.state.log_history, f)

# Plot the loss
loss_values = [x["loss"] for x in trainer.state.log_history if "loss" in x]
steps = [x["step"] for x in trainer.state.log_history if "loss" in x]

if len(loss_values) > 1:
    plt.figure(figsize=(8, 5))
    plt.plot(steps, loss_values, label="Training Loss", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(LOG_DIR, "loss_plot.png")
    plt.savefig(plot_path)
else:
    print("No loss data found in trainer.state.log_history — skipping plot.")

# Clean up
del model, trainer, tokenizer, dataset, tokenized_dataset

gc.collect()
torch.cuda.empty_cache()
Accelerator().free_memory()
