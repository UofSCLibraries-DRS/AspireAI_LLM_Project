from dataclasses import dataclass, field
from typing import Dict, List
from fine_tuning.inference import InferenceResult
from bert_score import score

import os
import csv
from collections import defaultdict


# TODOs:
#   - Optimize KV by sub batching prompt formats
@dataclass
class InferenceResultWithMetrics(InferenceResult):
    metrics: Dict[str, float] = field(default_factory=dict)


def calculate_bertscore_batched(
    results: List[InferenceResult],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str = "cuda",
    batch_size: int = 64,
) -> List[InferenceResultWithMetrics]:
    """ """
    # Collect all candidates and references with tracking info
    all_candidates = []
    all_references = []
    result_mapping = []  # Maps back to (result_idx, response_idx)

    for result_idx, result in enumerate(results):
        for response_idx, response in enumerate(result.responses):
            all_candidates.append(response)
            all_references.append(result.ground_truth)
            result_mapping.append((result_idx, response_idx))

    if not all_candidates:
        # No responses to evaluate
        return [
            InferenceResultWithMetrics(**vars(result), metrics={}) for result in results
        ]

    # Calculate BERTScore for all at once
    P, R, F1 = score(
        cands=all_candidates,
        refs=all_references,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        lang="en",
        verbose=False,
    )

    # Convert to lists
    precision_scores = P.tolist()
    recall_scores = R.tolist()
    f1_scores = F1.tolist()

    # Organize scores back by result
    result_scores = {}
    for idx, (result_idx, response_idx) in enumerate(result_mapping):
        if result_idx not in result_scores:
            result_scores[result_idx] = {"precision": [], "recall": [], "f1": []}
        result_scores[result_idx]["precision"].append(precision_scores[idx])
        result_scores[result_idx]["recall"].append(recall_scores[idx])
        result_scores[result_idx]["f1"].append(f1_scores[idx])

    # Create results with metrics
    results_with_metrics = []
    for result_idx, result in enumerate(results):
        if result_idx not in result_scores:
            # No responses for this result
            results_with_metrics.append(
                InferenceResultWithMetrics(**vars(result), metrics={})
            )
            continue

        scores = result_scores[result_idx]
        f_scores = scores["f1"]

        metrics = {
            "bertscore": sum(f_scores) / len(f_scores),
        }

        results_with_metrics.append(
            InferenceResultWithMetrics(**vars(result), metrics=metrics)
        )

    return results_with_metrics


def save_inference_results_with_metrics(
    inference_results: List[InferenceResultWithMetrics],
):
    # Group by model and output file
    grouped = defaultdict(list)
    for result in inference_results:
        key = (result.model, result.output_file, result.prompt_template)
        grouped[key].append(result)

    # Track metrics by model-prompt combo for averaging
    model_prompt_metrics = defaultdict(lambda: defaultdict(list))

    # Write each group to the corresponding csv
    for (model, output_file, prompt_template), results in grouped.items():
        prompt_name = os.path.basename(prompt_template).removesuffix(".yaml")
        csv_path = os.path.join(model, "results", prompt_name, output_file)

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Collect all unique metric keys from all results in this group
        metric_keys = set()
        for result in results:
            metric_keys.update(result.metrics.keys())
            # Accumulate metrics for this model-prompt combo
            combo_key = (model, prompt_name)
            for metric_key, metric_value in result.metrics.items():
                model_prompt_metrics[combo_key][metric_key].append(metric_value)

        # Sort metric keys for consistent column ordering
        metric_keys = sorted(metric_keys)

        # Build fieldnames: base fields + metric fields
        fieldnames = ["question", "answer", "response"] + metric_keys

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()

            for result in results:
                for response in result.responses:
                    row = {
                        "question": result.prompt,
                        "answer": result.ground_truth,
                        "response": response,
                    }
                    # Add all metrics to the row
                    for metric_key in metric_keys:
                        row[metric_key] = result.metrics.get(metric_key, "")

                    writer.writerow(row)

    # Save average metrics for each model-prompt combo
    for (model, prompt_name), metrics_dict in model_prompt_metrics.items():
        summary_path = os.path.join(
            model, "results", prompt_name, "metrics_summary.csv"
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)

        # Calculate averages
        metric_averages = {}
        for metric_key, values in metrics_dict.items():
            if values:
                metric_averages[metric_key] = sum(values) / len(values)

        # Write summary file
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["metric", "average"])
            for metric_key in sorted(metric_averages.keys()):
                writer.writerow([metric_key, metric_averages[metric_key]])
