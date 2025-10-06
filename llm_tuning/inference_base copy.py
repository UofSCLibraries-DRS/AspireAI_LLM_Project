import argparse
import pandas as pd
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
)
import os
from pathlib import Path
import csv


# Format prompt for base Llama 3.1 (with system prompt)
def format_prompt(question: str, system_prompt_path: str) -> str:
    with open(system_prompt_path, "r") as f:
        prompt_def = yaml.safe_load(f)

    template = prompt_def.get("template", "")
    return template.format(user_prompt=question)


class StopOnSequence(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        # check if the last tokens match the stop sequence
        if len(input_ids[0]) >= len(self.stop_ids):
            if input_ids[0][-len(self.stop_ids) :].tolist() == self.stop_ids:
                return True
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV with columns: question, answer",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save CSV with model outputs",
    )
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of responses to generate per question",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant. Answer clearly and concisely.",
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float16
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device="cuda",
    )

    # Load CSV
    df = pd.read_csv(args.input_csv)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns")

    # Storage for multiple responses
    all_responses = [[] for _ in range(args.num_samples)]

    # Generate for each row

    stop_ids = tokenizer("</s>", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0].tolist()

    stopping_criteria = StoppingCriteriaList([StopOnSequence(stop_ids)])
    for q in df["question"]:
        prompt = format_prompt(q, system_prompt_path=args.system_prompt)

        for i in range(args.num_samples):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Strip the original prompt so only the model’s continuation remains
            assistant_reply = generated_text[len(prompt) :].strip()
            all_responses[i].append(assistant_reply)

    # Add responses as new columns
    for i, responses in enumerate(all_responses, start=1):
        df[f"response_{i}"] = responses

    # Save results
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        args.output_csv,
        index=False,
        quoting=csv.QUOTE_ALL,
    )
    print(
        f"✅ Results with {args.num_samples} responses per question saved to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
