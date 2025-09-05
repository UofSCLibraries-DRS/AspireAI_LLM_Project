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


# Load your transcripts
df = pd.read_csv("/work/jaaydin/100_manually_cleaned.csv")
df = df[["Cleaned Transcript"]].dropna()
dataset = Dataset.from_pandas(df.rename(columns={"Cleaned Transcript": "text"}))


# Load LLaMA 3.1 tokenizer
local_model_path = "/work/jaaydin/downloaded_folder"

tokenizer = AutoTokenizer.from_pretrained(local_model_path)


# Ensure padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=1024
    )


tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])


model = AutoModelForCausalLM.from_pretrained(
    local_model_path, torch_dtype=torch.float16, device_map="auto"
)


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./llama3_1-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_strategy="epoch",
    fp16=True,
    evaluation_strategy="no",
    push_to_hub=False,
)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


trainer.train()
trainer.save_model("./llama3_1-mccray-lora-100")
tokenizer.save_pretrained("./llama3_1-mccray-lora-100")
