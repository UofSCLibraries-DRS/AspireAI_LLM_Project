from typing import Dict
from dataclasses import dataclass, field
from gaico import Experiment
from typing import List
from fine_tuning.inference import InferenceResult
import pandas as pd

import os
import csv
from collections import defaultdict

DELIMITER = "__@__"


@dataclass
class InferenceResultWithMetrics(InferenceResult):
    metrics: Dict[str, float] = field(default_factory=dict)


def _get_id(inference_result: InferenceResult, variant: str) -> str:
    return os.path.join(
        inference_result.model,
        "results",
        os.path.basename(inference_result.prompt_template).removesuffix(".yaml"),
        f"metrics_summary_{variant}.csv",
    )


def gaico_accuracy(
    results: List[InferenceResult],
) -> None:
    # Define ground truth variants to process
    variants = ["short", "ideal", "short_agg", "ideal_agg"]

    for variant in variants:
        ground_truth_attr = f"ground_truth_{variant}"

        # Group results by ground truth variant, filtering out empty/NaN values
        grouped = defaultdict(list)
        for result in results:
            gt_value = getattr(result, ground_truth_attr, None)

            # Skip if ground truth is None, empty string, or NaN
            if (
                gt_value is None
                or gt_value == ""
                or (isinstance(gt_value, float) and pd.isna(gt_value))
            ):
                continue

            grouped[gt_value].append(result)

        # Skip this variant if no valid ground truths found
        if not grouped:
            continue

        master_df = pd.DataFrame()

        for ground_truth, group_results in grouped.items():
            # Populate responses for exp class
            responses = {}
            for result in group_results:
                for i, response in enumerate(result.responses):
                    responses[f"{_get_id(result, variant)}{DELIMITER}{i}"] = response

            exp = Experiment(
                llm_responses=responses,
                reference_answer=ground_truth,
            )
            results_df = exp.compare(plot=False)

            # Append to master df
            master_df = pd.concat([master_df, results_df], ignore_index=True)

        # Split model_name to extract the original id and response index
        master_df[["original_id", "response_idx"]] = master_df["model_name"].str.split(
            DELIMITER,
            expand=True,
        )

        # Group by original_id and metric_name, then aggregate scores (average)
        aggregated = (
            master_df.groupby(["original_id", "metric_name"])["score"]
            .mean()
            .reset_index()
        )

        # Build output list and write summary files
        for original_id in aggregated["original_id"].unique():
            # Get metrics for this path
            metrics_subset = aggregated[aggregated["original_id"] == original_id]

            # Build metrics_dict
            metric_averages = {}
            for _, row in metrics_subset.iterrows():
                metric_averages[row["metric_name"]] = row["score"]

            # Write summary file - original_id is already the full path
            summary_path = original_id
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)

            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow(["metric", "average"])
                for metric_key in sorted(metric_averages.keys()):
                    writer.writerow([metric_key, metric_averages[metric_key]])
