import yaml
from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


@dataclass
class InferenceJob:
    group_id: str
    prompt_template: str
    prompt: str
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
def format_prompt(user_prompt: str, system_prompt_path: str, tokenizer=None) -> str:
    with open(system_prompt_path, "r") as f:
        prompt_def = yaml.safe_load(f)

    template = prompt_def.get("template", "")

    if tokenizer is not None and tokenizer.eos_token is not None:
        template = template.replace("</s>", tokenizer.eos_token)

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
            tokenizer=tokenizer,
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

            result.responses.append(response)

        all_results.append(result)

    return all_results
