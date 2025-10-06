import csv
import os
import yaml
from dataclasses import dataclass
from typing import List
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


@dataclass
class InferenceJob:
    group_id: str
    prompt_template: str
    prompt: str
    ground_truth: str
    stop_sequence: str | None = None
    repeat_param: int = 1
    max_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: float = 50


@dataclass
class InferenceResult:
    job: InferenceJob
    responses: List[str]


# Format prompt for base Llama 3.1 (with system prompt)
def format_prompt(user_prompt: str, system_prompt_path: str) -> str:
    with open(system_prompt_path, "r") as f:
        prompt_def = yaml.safe_load(f)

    template = prompt_def.get("template", "")

    return template.format(user_prompt=user_prompt)


def inference(
    model_dir: str,
    jobs: List[InferenceJob],
) -> List[InferenceResult]:
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device="cuda",
    )

    all_results: List[InferenceResult] = []

    for job in jobs:
        prompt = format_prompt(
            user_prompt=job.prompt,
            system_prompt_path=job.prompt_template,
        )

        result = InferenceResult(job=job, responses=[])

        for _ in range(job.repeat_param):
            output = generator(
                prompt,
                max_new_tokens=job.max_tokens,
                temperature=job.temperature,
                do_sample=True,
                top_p=job.top_p,
                top_k=job.top_k,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_text = output[0]["generated_text"]

            # Strip the original prompt so only the model’s continuation remains
            response = generated_text[len(prompt) :].strip()

            if job.stop_sequence:
                if job.stop_sequence in response:
                    response = response.split(job.stop_sequence, 1)[0].strip()

            result.responses.append(response)

        all_results.append(result)

    try:
        # Delete model-related objects
        del model
        del tokenizer
        del generator
        torch.cuda.empty_cache()  # clears cached memory
        print(f"🧹 Freed GPU memory for {model_dir}")
    except Exception as e:
        print(f"Warning: could not fully free model for {model_dir}: {e}")

    return all_results


def main():
    question_csv = "/work/jaaydin/data/combined.csv"
    df = pd.read_csv(question_csv)

    prompt_templates = [
        "config/prompts/base_p1.yaml",
        "config/prompts/base_p2.yaml",
        "config/prompts/base_p3.yaml",
        "config/prompts/base_p4.yaml",
        "config/prompts/base_p5.yaml",
        "config/prompts/icl_test/icl_p6.yaml",
        "config/prompts/icl_test/icl_p7.yaml",
        "config/prompts/icl_test/icl_p8.yaml",
    ]
    stop_sequences = [
        "\nQuestion:",
        "\nQuestion:",
        "\nQuestion:",
        "\nQuestion:",
        "\nQuestion:",
        "\nQuestion:",
        "\nQuestion:",
        "</s>",
    ]

    model_dirs = [
        "/work/jaaydin/models/Llama-3.1-8B",
        "/work/jaaydin/outputs/1000_mccray_dirty_base_Llama_8b/model",
        "/work/jaaydin/outputs/100_mccray_clean_base_Llama_8b/model",
        "/work/jaaydin/outputs/100_mccray_dirty_base_Llama_8b/model",
    ]

    jobs: List[InferenceJob] = []

    for prompt_template, stop_sequence in zip(prompt_templates, stop_sequences):
        for _, row in df.iterrows():
            job = InferenceJob(
                group_id=0,
                prompt_template=prompt_template,
                prompt=row["question"],
                ground_truth=row["answer"],
                stop_sequence=stop_sequence,
                repeat_param=5,
            )
            jobs.append(job)

    for model_dir in model_dirs:
        results = inference(model_dir=model_dir, jobs=jobs)

        output_dir = os.path.join(model_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        grouped_results: dict[str, list] = {}
        for result in results:
            prompt_name = os.path.splitext(
                os.path.basename(result.job.prompt_template)
            )[0]
            grouped_results.setdefault(prompt_name, []).append(result)

        for prompt_name, prompt_results in grouped_results.items():
            rows = []
            for result in prompt_results:
                responses = result.responses + [""] * (5 - len(result.responses))
                row = {
                    "question": result.job.prompt,
                    "answer": result.job.ground_truth,
                    "response_1": responses[0],
                    "response_2": responses[1],
                    "response_3": responses[2],
                    "response_4": responses[3],
                    "response_5": responses[4],
                }
                rows.append(row)

            result_df = pd.DataFrame(rows)

            # Save CSV with full quoting
            csv_path = os.path.join(output_dir, f"{prompt_name}_results.csv")
            result_df.to_csv(
                csv_path,
                index=False,
                quoting=csv.QUOTE_ALL,
            )
            print(f"Saved {model_dir} --- {prompt_name} results")


if __name__ == "__main__":
    main()
