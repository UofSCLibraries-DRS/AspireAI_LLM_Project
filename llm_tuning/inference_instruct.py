import argparse
import csv
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from pathlib import Path


# Format for Llama 3.1 chat-style prompt
def format_prompt(question: str) -> str:
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"


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
    args = parser.parse_args()

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, device_map="auto"
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

    generated_answers = []

    # Generate for each row
    for q in df["question"]:
        prompt = format_prompt(q)
        output = generator(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_p=0.9,
        )
        generated_text = output[0]["generated_text"]

        # Remove the prompt part so only the assistant's response remains
        assistant_reply = generated_text.split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[-1].strip()
        generated_answers.append(assistant_reply)

    # Save results
    df["generated_answer"] = generated_answers
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        args.output_csv,
        index=False,
        quoting=csv.QUOTE_ALL,
    )

    print(f"✅ Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
