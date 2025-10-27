import os
import yaml
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_TEMPERATURE = os.getenv("MODEL_TEMPERATURE")
PROMPT_TEMPLATE_PATH = os.getenv("PROMPT_TEMPLATE_PATH")

assert MODEL_PATH, "MODEL_PATH is not set in .env"
assert PROMPT_TEMPLATE_PATH, "PROMPT_TEMPLATE_PATH is not set in .env"

# Load template config
with open(PROMPT_TEMPLATE_PATH, "r") as f:
    config = yaml.safe_load(f)

TEMPLATE = config.get("template", "")
STOP_SEQUENCES = config.get("stop_sequences", [])

# Initialize model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto")

app = FastAPI()

# Apply stop sequences
def post_process(text: str, templated_prompt: str) -> str:
    # Remove prompt prefix
    text = text.removeprefix(templated_prompt)

    min_idx = len(text)
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            min_idx = min(idx, min_idx)
    text = text[:min_idx].strip()
    return text

@app.post("/generate")
async def generate(prompt: str, max_new_tokens: int = 128):
    # Apply prompt template
    templated_prompt = TEMPLATE.format(user_prompt=prompt)

    # Tokenize and generate
    inputs = tokenizer(templated_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
        )

    # Decode output
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    trimmed = post_process(full_text, templated_prompt)

    return {"text": trimmed}

