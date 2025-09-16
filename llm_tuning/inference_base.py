import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from pathlib import Path


# Format prompt for base Llama 3.1 (with system prompt)
def format_prompt(question: str, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = "You are a helpful assistant. Answer clearly and concisely."
    return f"{system_prompt}\n\nQuestion: {question}\nAnswer:"


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
	device="cuda"
    )

    # Load CSV
    df = pd.read_csv(args.input_csv)
    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must contain 'question' and 'answer' columns")

    # Storage for multiple responses
    all_responses = [[] for _ in range(args.num_samples)]

    # Generate for each row
    for q in df["question"]:
        prompt = format_prompt(q, system_prompt=args.system_prompt)

        for i in range(args.num_samples):
            output = generator(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_text = output[0]["generated_text"]

            # Strip the original prompt so only the model’s continuation remains
            assistant_reply = generated_text[len(prompt) :].strip()
            all_responses[i].append(assistant_reply)

    # Add responses as new columns
    for i, responses in enumerate(all_responses, start=1):
        df[f"response_{i}"] = responses

    # Save results
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(
        f"✅ Results with {args.num_samples} responses per question saved to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
