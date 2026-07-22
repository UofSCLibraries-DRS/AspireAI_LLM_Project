#!/usr/bin/env python3
"""Combine per-model metrics summaries into four analysis CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path


DEFAULT_RESULTS_DIR = Path.home() / "Research" / "library" / "data" / "results"
DEFAULT_OUTPUT_DIR_NAME = "unified_metrics_summaries"

SUMMARY_FILES = {
    "short": "metrics_summary_short.csv",
    "ideal": "metrics_summary_ideal.csv",
    "short_agg": "metrics_summary_short_agg.csv",
    "ideal_agg": "metrics_summary_ideal_agg.csv",
}

METADATA_FIELDS = [
    "model_path",
    "model_folder",
    "model_name",
    "uses_metadata",
    "epochs",
    "sft_size",
    "training_cfg",
    "lora_rank",
    "prompt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unify downloaded model metrics_summary CSVs by variant.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(os.environ.get("RESULTS_DIR", DEFAULT_RESULTS_DIR)),
        help=f"Root containing model result folders. Default: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the four unified CSVs. "
            f"Default: RESULTS_DIR/{DEFAULT_OUTPUT_DIR_NAME}"
        ),
    )
    return parser.parse_args()


def parse_model_folder(folder_name: str) -> dict[str, str]:
    """Parse names like M11.E20.NQ2k.03.r16 into analysis attributes."""
    parts = folder_name.split(".")
    attrs = {
        "model_folder": folder_name,
        "model_name": parts[0],
        "uses_metadata": "false",
        "epochs": "",
        "sft_size": "",
        "training_cfg": "",
        "lora_rank": "",
    }

    remaining = parts[1:]
    if "meta" in remaining:
        attrs["uses_metadata"] = "true"
        remaining = [part for part in remaining if part != "meta"]

    if remaining and re.fullmatch(r"E\d+", remaining[0]):
        attrs["epochs"] = remaining.pop(0)

    if remaining:
        attrs["sft_size"] = remaining.pop(0)
    if remaining:
        attrs["training_cfg"] = remaining.pop(0)
    if remaining:
        attrs["lora_rank"] = remaining.pop(0)

    return attrs


def read_metrics(summary_path: Path) -> dict[str, str]:
    metrics: dict[str, str] = {}
    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric", "").strip()
            if metric:
                metrics[metric] = row.get("average", "").strip()
    return metrics


def collect_rows(results_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows_by_variant = {variant: [] for variant in SUMMARY_FILES}

    for variant, filename in SUMMARY_FILES.items():
        for summary_path in sorted(results_dir.rglob(filename)):
            prompt_dir = summary_path.parent
            prompt_root = prompt_dir.parent
            if prompt_root.name != "results":
                continue

            model_dir = prompt_root.parent
            model_attrs = parse_model_folder(model_dir.name)
            model_path = model_dir.relative_to(results_dir).as_posix()
            row = {
                "model_path": model_path,
                **model_attrs,
                "prompt": prompt_dir.name,
                **read_metrics(summary_path),
            }
            rows_by_variant[variant].append(row)

    return rows_by_variant


def write_unified_csvs(
    rows_by_variant: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant, rows in rows_by_variant.items():
        metric_fields = sorted(
            {
                key
                for row in rows
                for key in row
                if key not in METADATA_FIELDS
            }
        )
        fieldnames = METADATA_FIELDS + metric_fields
        output_path = output_dir / f"metrics_summary_{variant}_unified.csv"

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / DEFAULT_OUTPUT_DIR_NAME
    )

    if not results_dir.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    rows_by_variant = collect_rows(results_dir)
    write_unified_csvs(rows_by_variant, output_dir)


if __name__ == "__main__":
    main()
