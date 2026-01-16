import csv
import gc
import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict


# TODOs:
#   - Optimize KV by sub batching prompt formats
@dataclass
class InferenceJob:
    output_file: str
    model: str
    prompt_template: str  # Path to .yaml file containing the prompt template
    prompt: str
    ground_truth: str
    stop_sequences: List[str] = field(
        default_factory=list
    )  # Internal logic will fill this field from the prompt template
    repeat_param: int = 5
    max_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: float = 50


@dataclass
class InferenceResult(InferenceJob):
    responses: List[str] = field(default_factory=list)


# Format prompt for base Llama 3.1 (with system prompt)
def _handle_prompt_formatting(
    user_prompt: str, system_prompt_path: str
) -> Tuple[str, List[str]]:
    with open(system_prompt_path, "r") as f:
        prompt_def = yaml.safe_load(f)

    template = prompt_def.get("template", "")
    stop_sequences = prompt_def.get("stop_sequences", [])

    return template.format(user_prompt=user_prompt), stop_sequences


# Helper collate
def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompts = [x["prompt"] for x in batch]
    idxs = [x["idx"] for x in batch]
    jobs_batch = [x["job"] for x in batch]
    return {"prompts": prompts, "idxs": idxs, "jobs": jobs_batch}


def _batched_inference(
    jobs: List[InferenceJob],
    device: str = "cuda",
    batch_size: int = 16,
    dtype: torch.dtype = torch.float32,
    max_prompt_length=1024,
) -> List[InferenceResult]:
    # Prepare prompts for each job
    prepared = []
    for i, job in enumerate(jobs):
        prompt_text, stop_sequences = _handle_prompt_formatting(
            user_prompt=job.prompt, system_prompt_path=job.prompt_template
        )
        job.stop_sequences = stop_sequences
        prepared.append({"idx": i, "prompt": prompt_text, "job": job})

    # Build groups: map key -> list[item]
    def group_key(item: Dict[str, Any]) -> Tuple:
        j: InferenceJob = item["job"]
        return (j.model, j.max_tokens, j.temperature, j.top_p, j.top_k)

    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for item in prepared:
        k = group_key(item)
        groups.setdefault(k, []).append(item)

    # Prepare empty results
    all_results: List[InferenceResult] = [
        InferenceResult(**vars(j), responses=[]) for j in jobs
    ]

    for key, items in groups.items():
        # Unwrap inference params from key
        model_path, max_new_tokens, temperature, top_p, top_k = key

        # Load model & tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=True, padding_side="left"
        )
        tokenizer.model_max_length = max_prompt_length

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
        )
        if device == "cuda":
            model.to("cuda")

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
                expanded_job_refs: List[InferenceJob] = []
                expanded_original_idxs: List[int] = []
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
                    for stop_sequence in job_ref.stop_sequences:
                        if stop_sequence in cont:
                            cont = cont.split(stop_sequence, 1)[0].strip()

                    all_results[orig_idx].responses.append(cont)

                # small cleanup between batches
                if device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        try:
            del model, tokenizer
        except Exception:
            pass
    # final cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return all_results


def batched_inference(
    jobs: List[InferenceJob],
    device: str = "cuda",
    batch_size: int = 16,
    max_prompt_length=1024,
):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return _batched_inference(
        jobs,
        device,
        batch_size,
        dtype,
        max_prompt_length,
    )


def save_inference_results(inference_results: List[InferenceResult]):
    # Group by model and output file
    grouped = defaultdict(list)
    for result in inference_results:
        key = (result.model, result.output_file, result.prompt_template)
        grouped[key].append(result)

    # Append each response to the corresponding csv
    for (model, output_file, prompt_template), results in grouped.items():
        prompt_name = os.path.basename(prompt_template).removesuffix(".yaml")
        csv_path = os.path.join(model, "results", prompt_name, output_file)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        fieldnames = ["question", "answer", "response"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()

            for result in results:
                for response in result.responses:
                    writer.writerow(
                        {
                            "question": result.prompt,
                            "answer": result.ground_truth,
                            "response": response,
                        }
                    )
