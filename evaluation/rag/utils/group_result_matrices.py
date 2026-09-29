#!/usr/bin/env python3
"""Group model results by RAG type and fine-tuning status.

Two CSV matrices and two color-coded figures per score subset are produced.
The value column defaults to ranking_score and can be changed to another
numeric column such as mean_score.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


DEFAULT_INPUT = Path("results/M14_greedy/model_rankings.csv")
DEFAULT_AVERAGE_OUTPUT = Path("results/M14_greedy/rag_finetune_average.csv")
DEFAULT_MAX_OUTPUT = Path("results/M14_greedy/rag_finetune_max.csv")
DEFAULT_FIGURES_DIR = Path(
    "results/M14_greedy/figures/rag_vs_fine-tuning"
)
TUNING_COLUMNS = ("fine_tuned", "not_fine_tuned")
AVERAGE_OUTPUT_FIELDS = ("subset", "rag_type", *TUNING_COLUMNS)
MAX_OUTPUT_FIELDS = (
    "subset",
    "rag_type",
    "fine_tuned",
    "fine_tuned_model",
    "not_fine_tuned",
    "not_fine_tuned_model",
)
RAG_SUFFIX_PATTERN = re.compile(r"^(?P<model>.+?)(?: \((?P<rag>[^()]+)\))?$")
FINE_TUNED_PATTERN = re.compile(r"\.e\d+\.(?:MLP|no_MLP)\.r\d+$")


@dataclass(frozen=True)
class ModelValue:
    subset: str
    rag_type: str
    tuning_status: str
    model_id: str
    value: float


MatrixRow = dict[str, str | float]


def classify_model(model_id: str) -> tuple[str, str]:
    """Return the RAG type and tuning status for an evaluation model ID."""
    match = RAG_SUFFIX_PATTERN.fullmatch(model_id.strip())
    if match is None:
        raise ValueError(f"Unable to parse model ID: {model_id!r}")

    base_model = match.group("model")
    rag_type = match.group("rag") or "none"
    tuning_status = (
        "fine_tuned"
        if FINE_TUNED_PATTERN.search(base_model)
        else "not_fine_tuned"
    )
    return rag_type, tuning_status


def load_model_values(
    input_path: Path,
    value_column: str = "ranking_score",
) -> list[ModelValue]:
    """Load one scalar value per model and score subset."""
    with input_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_path}")

        required = {"subset", "model_id", value_column}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(
                f"Input CSV is missing column(s): {', '.join(missing)}"
            )

        model_values = []
        for line_number, row in enumerate(reader, start=2):
            value = _finite_float(row.get(value_column))
            if value is None:
                continue

            model_id = row.get("model_id", "").strip()
            if not model_id:
                raise ValueError(f"Missing model_id on line {line_number}")
            rag_type, tuning_status = classify_model(model_id)
            model_values.append(
                ModelValue(
                    subset=row.get("subset", "").strip(),
                    rag_type=rag_type,
                    tuning_status=tuning_status,
                    model_id=model_id,
                    value=value,
                )
            )

    if not model_values:
        raise ValueError(
            f"Input CSV contains no finite {value_column}: {input_path}"
        )
    return model_values


def build_average_matrix(model_values: Sequence[ModelValue]) -> list[MatrixRow]:
    """Average model values within each RAG/fine-tuning cell."""
    grouped = _group_model_values(model_values)
    rows = []
    for subset, rag_type in _sorted_group_keys(grouped):
        row: MatrixRow = {"subset": subset, "rag_type": rag_type}
        for tuning_status in TUNING_COLUMNS:
            values = grouped[(subset, rag_type)][tuning_status]
            row[tuning_status] = (
                sum(item.value for item in values) / len(values)
                if values
                else ""
            )
        rows.append(row)
    return rows


def build_max_matrix(model_values: Sequence[ModelValue]) -> list[MatrixRow]:
    """Select the highest-scoring model in each RAG/fine-tuning cell."""
    grouped = _group_model_values(model_values)
    rows = []
    for subset, rag_type in _sorted_group_keys(grouped):
        row: MatrixRow = {"subset": subset, "rag_type": rag_type}
        for tuning_status in TUNING_COLUMNS:
            values = grouped[(subset, rag_type)][tuning_status]
            if not values:
                row[tuning_status] = ""
                row[f"{tuning_status}_model"] = ""
                continue
            winner = sorted(
                values,
                key=lambda item: (-item.value, item.model_id),
            )[0]
            row[tuning_status] = winner.value
            row[f"{tuning_status}_model"] = winner.model_id
        rows.append(row)
    return rows


def _group_model_values(
    model_values: Sequence[ModelValue],
) -> dict[tuple[str, str], dict[str, list[ModelValue]]]:
    grouped: dict[tuple[str, str], dict[str, list[ModelValue]]] = defaultdict(
        lambda: {column: [] for column in TUNING_COLUMNS}
    )
    for model_value in model_values:
        grouped[(model_value.subset, model_value.rag_type)][
            model_value.tuning_status
        ].append(model_value)
    return dict(grouped)


def _sorted_group_keys(
    grouped: dict[tuple[str, str], object],
) -> list[tuple[str, str]]:
    return sorted(
        grouped,
        key=lambda key: (key[0], key[1] != "none", key[1]),
    )


def write_matrix(
    rows: Sequence[MatrixRow],
    output_path: Path,
    fieldnames: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(
    average_rows: Sequence[MatrixRow],
    max_rows: Sequence[MatrixRow],
    figures_dir: Path,
    value_label: str,
) -> list[Path]:
    """Write average and maximum heatmaps for every score subset."""
    output_paths = []
    subsets = sorted({str(row["subset"]) for row in average_rows})
    for subset in subsets:
        subset_dir = figures_dir / _figure_subset_name(subset)
        subset_dir.mkdir(parents=True, exist_ok=True)
        average_path = subset_dir / "average.png"
        max_path = subset_dir / "max.png"
        _write_heatmap(
            rows=[row for row in average_rows if row["subset"] == subset],
            output_path=average_path,
            title=f"{subset}: average {value_label}",
            show_models=False,
        )
        _write_heatmap(
            rows=[row for row in max_rows if row["subset"] == subset],
            output_path=max_path,
            title=f"{subset}: maximum {value_label}",
            show_models=True,
        )
        output_paths.extend((average_path, max_path))
    return output_paths


def _write_heatmap(
    rows: Sequence[MatrixRow],
    output_path: Path,
    title: str,
    show_models: bool,
) -> None:
    rag_types = [str(row["rag_type"]) for row in rows]
    values = [
        [
            _finite_float(row.get(tuning_status)) or 0.0
            for tuning_status in TUNING_COLUMNS
        ]
        for row in rows
    ]
    finite_values = [value for row_values in values for value in row_values]
    norm = _color_norm(finite_values)

    figure_height = max(4.5, len(rag_types) * 0.9)
    fig, ax = plt.subplots(figsize=(11 if show_models else 7, figure_height))
    image = ax.imshow(values, cmap="RdYlGn", norm=norm, aspect="auto")
    ax.set_xticks(range(len(TUNING_COLUMNS)), labels=TUNING_COLUMNS)
    ax.set_yticks(range(len(rag_types)), labels=rag_types)
    ax.set_xlabel("Fine-tuning status")
    ax.set_ylabel("RAG type")
    ax.set_title(title)

    for row_index, row in enumerate(rows):
        for column_index, tuning_status in enumerate(TUNING_COLUMNS):
            value = _finite_float(row.get(tuning_status))
            if value is None:
                label = "n/a"
            elif show_models:
                model = str(row.get(f"{tuning_status}_model", ""))
                label = f"{value:.3f}\n{_short_model_label(model)}"
            else:
                label = f"{value:.3f}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7 if show_models else 9,
            )

    fig.colorbar(image, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _color_norm(values: Sequence[float]) -> Normalize:
    minimum = min(values)
    maximum = max(values)
    if minimum < 0.0 < maximum:
        return TwoSlopeNorm(vmin=minimum, vcenter=0.0, vmax=maximum)
    if math.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.1, 0.1)
        return Normalize(vmin=minimum - padding, vmax=maximum + padding)
    return Normalize(vmin=minimum, vmax=maximum)


def _short_model_label(model_id: str) -> str:
    match = RAG_SUFFIX_PATTERN.fullmatch(model_id)
    base_model = match.group("model") if match is not None else model_id
    if base_model.startswith("M14.meta."):
        base_model = base_model.removeprefix("M14.meta.")
    elif base_model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        base_model = "base"
    return "\n".join(textwrap.wrap(base_model, width=24))


def _figure_subset_name(subset: str) -> str:
    name = subset.removeprefix("answer_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "unknown"


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
        "--value-column",
        default="ranking_score",
        help="Scalar column to aggregate (default: ranking_score)",
    )
    parser.add_argument(
        "--average-output",
        type=Path,
        default=DEFAULT_AVERAGE_OUTPUT,
        help=f"Average matrix path (default: {DEFAULT_AVERAGE_OUTPUT})",
    )
    parser.add_argument(
        "--max-output",
        type=Path,
        default=DEFAULT_MAX_OUTPUT,
        help=f"Maximum matrix path (default: {DEFAULT_MAX_OUTPUT})",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Figure root directory (default: {DEFAULT_FIGURES_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model_values = load_model_values(args.input, args.value_column)
        average_rows = build_average_matrix(model_values)
        max_rows = build_max_matrix(model_values)
        write_matrix(
            average_rows,
            args.average_output,
            AVERAGE_OUTPUT_FIELDS,
        )
        write_matrix(max_rows, args.max_output, MAX_OUTPUT_FIELDS)
        figure_paths = write_figures(
            average_rows,
            max_rows,
            args.figures_dir,
            args.value_column,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Wrote average matrix to {args.average_output}")
    print(f"Wrote maximum matrix to {args.max_output}")
    for figure_path in figure_paths:
        print(f"Wrote figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
