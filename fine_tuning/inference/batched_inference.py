import csv
import gc
import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# TODOs:
#   - Optimize KV by sub batching prompt formats
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
def _format_prompt(user_prompt: str, system_prompt_path: str) -> str:
    with open(system_prompt_path, "r") as f:
        prompt_def = yaml.safe_load(f)

    template = prompt_def.get("template", "")

    return template.format(user_prompt=user_prompt)


# Helper collate
def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompts = [x["prompt"] for x in batch]
    idxs = [x["idx"] for x in batch]
    jobs_batch = [x["job"] for x in batch]
    return {"prompts": prompts, "idxs": idxs, "jobs": jobs_batch}


def batched_inference(
    model_dir: str,
    jobs: List[InferenceJob],
    device: str = "cuda",
    batch_size: int = 16,
    dtype: torch.dtype = torch.float32,
    use_fp16_if_available: bool = False,
    max_prompt_length=1024,
) -> List[InferenceResult]:
    if use_fp16_if_available and torch.cuda.is_available():
        model_dtype = (
            torch.float16
            if dtype == torch.float16 or use_fp16_if_available
            else torch.float32
        )
    else:
        model_dtype = torch.float32

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
    )
    tokenizer.model_max_length = max_prompt_length

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=model_dtype,
    )
    if device == "cuda":
        model.to("cuda")

    # Prepare prompts for each job
    prepared = []
    for i, job in enumerate(jobs):
        prompt_text = _format_prompt(
            user_prompt=job.prompt, system_prompt_path=job.prompt_template
        )
        prepared.append({"idx": i, "prompt": prompt_text, "job": job})

    # Build groups: map key -> list[item]
    def group_key(item: Dict[str, Any]) -> Tuple:
        j: InferenceJob = item["job"]
        return (j.max_tokens, j.temperature, j.top_p, j.top_k, j.stop_sequence)

    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for item in prepared:
        k = group_key(item)
        groups.setdefault(k, []).append(item)

    # Prepare empty results
    all_results: List[InferenceResult] = [
        InferenceResult(job=j, responses=[]) for j in jobs
    ]

    for key, items in groups.items():
        # Unwrap inference params from key
        max_new_tokens, temperature, top_p, top_k, stop_sequence = key

        dataloader = DataLoader(
            items, batch_size=batch_size, collate_fn=_collate_fn, shuffle=False
        )

        with torch.no_grad():
            for batch in dataloader:
                prompts: List[str] = batch["prompts"]
                idxs: List[int] = batch["idxs"]
                jobs_batch: List[InferenceJob] = batch["jobs"]

                # Tokenize batch (pad to longest)
                tokenized = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    pad_to_multiple_of=32,
                )
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]
                input_lengths = (
                    (attention_mask != 0).sum(dim=1).tolist()
                )  # length per example

                # Move to device
                if device == "cuda":
                    input_ids = input_ids.to("cuda")
                    attention_mask = attention_mask.to("cuda")

                # Expand per repeat_param
                expanded_input_ids = []
                expanded_attention_mask = []
                expanded_input_lens = []
                expanded_job_refs = []
                expanded_original_idxs = []
                for i_local, job in enumerate(jobs_batch):
                    rc = job.repeat_param
                    for _ in range(rc):
                        expanded_input_ids.append(input_ids[i_local])
                        expanded_attention_mask.append(attention_mask[i_local])
                        expanded_input_lens.append(input_lengths[i_local])
                        expanded_job_refs.append(job)
                        expanded_original_idxs.append(idxs[i_local])

                input_ids = torch.stack(expanded_input_ids, dim=0)
                attention_mask = torch.stack(expanded_attention_mask, dim=0)
                batch_job_refs = expanded_job_refs
                batch_original_idxs = expanded_original_idxs

                # Generate
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id,
                    use_cache=True,
                )

                # The returned `generated_ids` is usually concatenation of input + generated.
                # We slice off the input portion using input lengths per example.
                # generated_ids shape: (batch_expanded, seq_len_generated_total)
                # For robustness, we'll compute the slice per-row.
                decoded_continuations: List[str] = []
                gen_ids_cpu = generated_ids.cpu()
                for gen_row in gen_ids_cpu:
                    text = tokenizer.decode(gen_row, skip_special_tokens=True).strip()
                    decoded_continuations.append(text)

                # Append to results with stop sequence handling
                for continuation, job_ref, orig_idx in zip(
                    decoded_continuations, batch_job_refs, batch_original_idxs
                ):
                    cont = continuation

                    # Remove prompt from output
                    full_prompt = prepared[orig_idx]["prompt"]
                    cont = cont.removeprefix(full_prompt)

                    # truncate at stop_sequence if present
                    if job_ref.stop_sequence and job_ref.stop_sequence in cont:
                        cont = cont.split(job_ref.stop_sequence, 1)[0].strip()

                    all_results[orig_idx].responses.append(cont)

                # small cleanup between batches
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
    # final cleanup
    try:
        del model, tokenizer
    except Exception:
        pass
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return all_results


def main():
    question_csv = "combined.csv"
    df = pd.read_csv(question_csv)

    prompt_templates = [
        "prompts/sft.yaml",
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
        # "gemma-3-270m_sft/model",
        "high_cpt_then_sft/model"
    ]

    jobs: List[InferenceJob] = []

    for prompt_template, stop_sequence in zip(prompt_templates, stop_sequences):
        for _, row in df.iterrows():
            job = InferenceJob(
                group_id=0,
                prompt_template=prompt_template,
                prompt=row["question"],
                ground_truth=row["answer"],
                repeat_param=5,
                stop_sequence="<end_answer>",
            )
            jobs.append(job)

    for model_dir in model_dirs:
        results = batched_inference(model_dir=model_dir, jobs=jobs)

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
