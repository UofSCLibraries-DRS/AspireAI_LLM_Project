#!/usr/bin/env python3
"""Create per-subset model bar charts for every available metric."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from utils.group_result_matrices import classify_model
except ModuleNotFoundError:
    from group_result_matrices import classify_model


DEFAULT_INPUT = Path("results/M14_greedy/model_rankings.csv")
DEFAULT_OUTPUT_DIR = Path("results/M14_greedy/figures/metric_bar_charts")
METRIC_PREFIX = "mean_"
NON_METRIC_MEAN_FIELDS = {"mean_score"}


@dataclass(frozen=True)
class ConfigurationScores:
    subset: str
    model_id: str
    metrics: dict[str, float]

    @property
    def mean(self) -> float:
        return sum(self.metrics.values()) / len(self.metrics)


def load_configuration_scores(
    input_path: Path,
) -> tuple[list[ConfigurationScores], tuple[str, ...]]:
    """Load per-model metric means from a model-ranking CSV."""
    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_path}")

        required_fields = {"subset", "model_id"}
        missing_fields = sorted(required_fields - set(fieldnames))
        if missing_fields:
            raise ValueError(
                f"Input CSV is missing column(s): {', '.join(missing_fields)}"
            )

        metric_fields = [
            fieldname
            for fieldname in fieldnames
            if fieldname.startswith(METRIC_PREFIX)
            and fieldname not in NON_METRIC_MEAN_FIELDS
        ]
        if not metric_fields:
            raise ValueError(f"Input CSV has no per-metric mean columns: {input_path}")

        records = []
        for line_number, row in enumerate(reader, start=2):
            model_id = row.get("model_id", "").strip()
            if not model_id:
                raise ValueError(f"Missing model_id on line {line_number}")

            metrics = {
                fieldname.removeprefix(METRIC_PREFIX): value
                for fieldname in metric_fields
                if (value := _finite_float(row.get(fieldname))) is not None
            }
            if not metrics:
                continue
            records.append(
                ConfigurationScores(
                    subset=row.get("subset", "").strip(),
                    model_id=model_id,
                    metrics=metrics,
                )
            )

    if not records:
        raise ValueError(f"Input CSV contains no finite metric means: {input_path}")
    metric_names = tuple(
        fieldname.removeprefix(METRIC_PREFIX) for fieldname in metric_fields
    )
    return records, metric_names


def write_metric_bar_charts(
    records: Sequence[ConfigurationScores],
    metric_names: Sequence[str],
    output_dir: Path,
) -> list[Path]:
    """Write one chart per metric and one all-metric mean chart per subset."""
    output_paths = []
    subsets = sorted({record.subset for record in records})
    for subset in subsets:
        subset_records = [record for record in records if record.subset == subset]
        subset_dir = output_dir / _subset_directory_name(subset)
        subset_dir.mkdir(parents=True, exist_ok=True)

        for metric_name in metric_names:
            chart_records = [
                record for record in subset_records if metric_name in record.metrics
            ]
            if not chart_records:
                continue
            output_path = subset_dir / f"{_safe_filename(metric_name)}.png"
            _write_bar_chart(
                values=[
                    (record.model_id, record.metrics[metric_name])
                    for record in chart_records
                ],
                title=f"{subset}: {metric_name}",
                x_label=metric_name,
                output_path=output_path,
            )
            output_paths.append(output_path)

        configuration_mean_path = subset_dir / "configuration_mean.png"
        _write_bar_chart(
            values=[(record.model_id, record.mean) for record in subset_records],
            title=f"{subset}: mean across metrics per configuration",
            x_label="Mean metric value",
            output_path=configuration_mean_path,
        )
        output_paths.append(configuration_mean_path)
    return output_paths


def _write_bar_chart(
    values: Sequence[tuple[str, float]],
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    ordered_values = sorted(values, key=lambda item: (item[1], item[0]))
    rag_types = sorted({classify_model(model_id)[0] for model_id, _ in values})
    color_map = _rag_color_map(rag_types)

    figure_height = max(5.0, len(ordered_values) * 0.34)
    fig, ax = plt.subplots(figsize=(14, figure_height))
    bars = ax.barh(
        range(len(ordered_values)),
        [value for _, value in ordered_values],
        color=[
            color_map[classify_model(model_id)[0]] for model_id, _ in ordered_values
        ],
    )
    ax.set_yticks(
        range(len(ordered_values)),
        labels=[model_id for model_id, _ in ordered_values],
        fontsize=7,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Model / configuration")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.margins(x=0.12)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=6)
    ax.legend(
        handles=[
            Patch(facecolor=color_map[rag_type], label=rag_type)
            for rag_type in rag_types
        ],
        title="RAG type",
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _rag_color_map(rag_types: Sequence[str]) -> dict[str, object]:
    color_map = plt.get_cmap("tab10")
    return {
        rag_type: color_map(index % color_map.N)
        for index, rag_type in enumerate(rag_types)
    }


def _subset_directory_name(subset: str) -> str:
    return _safe_filename(subset.removeprefix("answer_"))


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "unknown"


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Model-ranking CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Chart output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records, metric_names = load_configuration_scores(args.input)
        output_paths = write_metric_bar_charts(
            records,
            metric_names,
            args.output_dir,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Wrote {len(output_paths)} metric bar charts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
