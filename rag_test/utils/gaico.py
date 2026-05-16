import csv
import os
import re
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from gaico import Experiment
from tqdm import tqdm

from utils.config import ExperimentConfig


GAICO_OUTPUT_DIRNAME = "gaico"
RESULT_FILENAME = "results.csv"
DEFAULT_METRIC_NAMES = [
    "Jaccard",
    "Cosine",
    "Levenshtein",
    "SequenceMatcher",
    "BLEU",
    "ROUGE",
    "JSD",
    "BERTScore",
]


def run_gaico(exp_config: ExperimentConfig) -> list[Path]:
    result_path = Path(exp_config.out) / RESULT_FILENAME
    if not result_path.exists():
        raise FileNotFoundError(f"Experiment results not found: {result_path}")

    rows, fieldnames = _load_results(result_path)
    output_dir = Path(exp_config.out) / GAICO_OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_ground_truth_columns(
        ground_truth_columns=exp_config.eval_data.ground_truth_columns,
        fieldnames=fieldnames,
        result_path=result_path,
    )

    progress_total = _count_scored_reference_groups(
        rows=rows,
        ground_truth_columns=exp_config.eval_data.ground_truth_columns,
    )
    output_paths = []
    with tqdm(
        total=progress_total,
        desc="Running Gaico",
        unit="group",
        disable=progress_total == 0,
    ) as progress:
        for ground_truth_column in exp_config.eval_data.ground_truth_columns:
            output_path = output_dir / f"{_safe_filename(ground_truth_column)}.csv"
            _write_ground_truth_scores(
                rows=rows,
                fieldnames=fieldnames,
                ground_truth_column=ground_truth_column,
                output_path=output_path,
                progress=progress,
            )
            output_paths.append(output_path)

    return output_paths


def _load_results(result_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with result_path.open("r", encoding="utf-8", newline="") as result_file:
        reader = csv.DictReader(result_file)
        if reader.fieldnames is None:
            raise ValueError(f"Results CSV is empty: {result_path}")
        return list(reader), reader.fieldnames


def _write_ground_truth_scores(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    ground_truth_column: str,
    output_path: Path,
    progress: tqdm | None = None,
) -> None:
    valid_groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    zero_score_rows = []

    for source_row_index, row in enumerate(rows):
        ground_truth_value = row.get(ground_truth_column, "")
        if _is_blank(ground_truth_value):
            continue

        if _is_blank(row.get("response", "")) or not _is_blank(row.get("error", "")):
            zero_score_rows.append((source_row_index, row))
            continue

        valid_groups[ground_truth_value].append((source_row_index, row))

    metric_scores_by_row: dict[int, dict[str, Any]] = {}
    metric_names = set()
    for ground_truth_value, group_rows in valid_groups.items():
        responses = {
            str(source_row_index): row["response"]
            for source_row_index, row in group_rows
        }
        experiment = Experiment(
            llm_responses=responses,
            reference_answer=ground_truth_value,
        )
        with _suppress_metric_output():
            comparison_df = experiment.compare(plot=False)
        if progress is not None:
            progress.update()

        row_by_index = {
            str(source_row_index): (source_row_index, row)
            for source_row_index, row in group_rows
        }
        for score_record in _iter_score_records(comparison_df):
            metric_name = score_record["metric_name"]
            metric_names.add(metric_name)

            source_row_index, _row = row_by_index[score_record["model_name"]]
            metric_scores_by_row.setdefault(source_row_index, {})[metric_name] = (
                score_record["score"]
            )

    zero_metric_names = sorted(metric_names) if metric_names else DEFAULT_METRIC_NAMES
    for source_row_index, _row in zero_score_rows:
        for metric_name in zero_metric_names:
            metric_scores_by_row.setdefault(source_row_index, {})[metric_name] = 0.0

    metric_fieldnames = sorted(metric_names) if metric_names else DEFAULT_METRIC_NAMES
    scored_rows = [
        _build_output_row(
            row=rows[source_row_index],
            metric_scores=metric_scores_by_row[source_row_index],
            metric_fieldnames=metric_fieldnames,
        )
        for source_row_index in sorted(metric_scores_by_row)
    ]
    _write_scores_csv(scored_rows, fieldnames, metric_fieldnames, output_path)


def _validate_ground_truth_columns(
    ground_truth_columns: list[str],
    fieldnames: list[str],
    result_path: Path,
) -> None:
    for ground_truth_column in ground_truth_columns:
        if ground_truth_column not in fieldnames:
            raise ValueError(
                f"Results CSV must contain configured ground truth column "
                f"`{ground_truth_column}`: {result_path}"
            )


def _count_scored_reference_groups(
    rows: list[dict[str, str]],
    ground_truth_columns: list[str],
) -> int:
    total = 0
    for ground_truth_column in ground_truth_columns:
        total += len(
            {
                row.get(ground_truth_column, "")
                for row in rows
                if not _is_blank(row.get(ground_truth_column, ""))
                and not _is_blank(row.get("response", ""))
                and _is_blank(row.get("error", ""))
            }
        )
    return total


def _iter_score_records(comparison_df: Any) -> list[dict[str, Any]]:
    records = []
    for record in comparison_df.to_dict("records"):
        records.append(
            {
                "model_name": str(record["model_name"]),
                "metric_name": str(record["metric_name"]),
                "score": record["score"],
            }
        )
    return records


def _build_output_row(
    row: dict[str, str],
    metric_scores: dict[str, Any],
    metric_fieldnames: list[str],
) -> dict[str, Any]:
    output_row = dict(row)
    for metric_name in metric_fieldnames:
        output_row[metric_name] = metric_scores.get(metric_name, "")
    return output_row


def _write_scores_csv(
    rows: list[dict[str, Any]],
    source_fieldnames: list[str],
    metric_fieldnames: list[str],
    output_path: Path,
) -> None:
    fieldnames = [*source_fieldnames, *metric_fieldnames]

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "ground_truth"


def _is_blank(value: str | None) -> bool:
    return value is None or not value.strip()


@contextmanager
def _suppress_metric_output():
    transformers_logging = None
    previous_transformers_verbosity = None
    try:
        from transformers.utils import logging as transformers_logging

        previous_transformers_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass

    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                yield
    finally:
        if transformers_logging is not None and previous_transformers_verbosity is not None:
            transformers_logging.set_verbosity(previous_transformers_verbosity)
