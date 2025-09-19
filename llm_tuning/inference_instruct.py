import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


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
    parser.add_argument("--num_samples", type=int, default=1, help="Number of responses per question")
    args = parser.parse_args()

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float16
    ).to("cuda")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=0,
    )

    # Load CSV
    df = pd.read_csv(args.input_csv)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns")

    # Generate multiple responses
    all_responses = {f"response_{i+1}": [] for i in range(args.num_samples)}

    for q in df["question"]:
        prompt = format_prompt(q)
        sample_responses = []
        for _ in range(args.num_samples):
            output = generator(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
            )
            generated_text = output[0]["generated_text"]
            assistant_reply = generated_text.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1].strip()
            sample_responses.append(assistant_reply)

        # Fill responses into columns
        for i, reply in enumerate(sample_responses):
            all_responses[f"response_{i+1}"].append(reply)

    # Attach responses to dataframe
    for col, values in all_responses.items():
        df[col] = values

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        args.output_csv,
        index=False,
        quoting=csv.QUOTE_ALL,
    )

    print(f"✅ Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()

