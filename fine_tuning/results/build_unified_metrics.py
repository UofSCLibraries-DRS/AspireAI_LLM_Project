#!/usr/bin/env python3
"""Build the unified metric summary CSVs from the result directory tree.

Each input is expected at:

    <model_path>/results/<prompt>/metrics_summary_*.csv

The input CSVs contain ``metric,average`` rows.  One wide output row is
created per input file, with model metadata inferred from the model folder.
For example, ``M12.meta.NQ2k.03.r16`` becomes M12, metadata=true, NQ2k,
configuration 03, and LoRA rank r16.

Run without arguments to rebuild the four unified CSVs next to this script.
Use ``--check`` to verify that the existing unified CSVs exactly represent
the current source tree without changing any files.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SOURCE_FILES = (
    "metrics_summary_ideal_agg.csv",
    "metrics_summary_ideal.csv",
    "metrics_summary_short_agg.csv",
    "metrics_summary_short.csv",
)

IDENTITY_COLUMNS = (
    "model_path",
    "model_folder",
    "model_name",
    "uses_metadata",
    "epochs",
    "sft_size",
    "training_cfg",
    "lora_rank",
    "prompt",
)

EPOCH_RE = re.compile(r"E\d+")


class AggregationError(Exception):
    """Raised when the result tree cannot be aggregated unambiguously."""


@dataclass(frozen=True)
class UnifiedSummary:
    source_name: str
    metric_columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]

    @property
    def output_name(self) -> str:
        return f"{Path(self.source_name).stem}_unified.csv"

    @property
    def columns(self) -> tuple[str, ...]:
        return IDENTITY_COLUMNS + self.metric_columns


def parse_model_folder(model_folder: str) -> dict[str, str]:
    """Extract model metadata from a dot-delimited result folder name."""
    tokens = model_folder.split(".")
    if not tokens[0] or any(not token for token in tokens):
        raise AggregationError(f"invalid model folder name: {model_folder!r}")

    model_name = tokens[0]
    uses_metadata = False
    epochs = ""
    positional_tokens: list[str] = []

    for token in tokens[1:]:
        if token == "meta":
            if uses_metadata:
                raise AggregationError(
                    f"duplicate metadata marker in model folder {model_folder!r}"
                )
            uses_metadata = True
        elif EPOCH_RE.fullmatch(token):
            if epochs:
                raise AggregationError(
                    f"multiple epoch markers in model folder {model_folder!r}"
                )
            epochs = token
        else:
            positional_tokens.append(token)

    if len(positional_tokens) > 3:
        raise AggregationError(
            f"too many unrecognized components in model folder {model_folder!r}; "
            "expected at most SFT size, training config, and LoRA rank"
        )

    positional_tokens.extend([""] * (3 - len(positional_tokens)))
    sft_size, training_cfg, lora_rank = positional_tokens
    return {
        "model_name": model_name,
        "uses_metadata": str(uses_metadata).lower(),
        "epochs": epochs,
        "sft_size": sft_size,
        "training_cfg": training_cfg,
        "lora_rank": lora_rank,
    }


def source_identity(root: Path, source_path: Path) -> dict[str, str]:
    """Build the identity fields for a metric summary source path."""
    relative = source_path.relative_to(root)
    result_indices = [
        index for index, component in enumerate(relative.parts) if component == "results"
    ]
    if len(result_indices) != 1:
        raise AggregationError(
            f"{relative}: expected exactly one 'results' path component"
        )

    results_index = result_indices[0]
    if results_index == 0:
        raise AggregationError(f"{relative}: missing model path before 'results'")

    prompt_parts = relative.parts[results_index + 1 : -1]
    if not prompt_parts:
        raise AggregationError(f"{relative}: missing prompt folder after 'results'")

    model_parts = relative.parts[:results_index]
    model_folder = model_parts[-1]
    identity = {
        "model_path": Path(*model_parts).as_posix(),
        "model_folder": model_folder,
        **parse_model_folder(model_folder),
        "prompt": Path(*prompt_parts).as_posix(),
    }
    return identity


def read_metric_table(source_path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    """Read and validate one long-form metric/average CSV."""
    try:
        handle = source_path.open(encoding="utf-8-sig", newline="")
    except OSError as error:
        raise AggregationError(f"could not read {source_path}: {error}") from error

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["metric", "average"]:
            raise AggregationError(
                f"{source_path}: expected columns ['metric', 'average'], "
                f"found {reader.fieldnames!r}"
            )

        metrics: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            metric = row.get("metric")
            average = row.get("average")
            if None in row or metric is None or average is None:
                raise AggregationError(
                    f"{source_path}:{line_number}: malformed metric row"
                )
            if not metric:
                raise AggregationError(
                    f"{source_path}:{line_number}: metric name is empty"
                )
            if metric in metrics:
                raise AggregationError(
                    f"{source_path}:{line_number}: duplicate metric {metric!r}"
                )
            if metric in IDENTITY_COLUMNS:
                raise AggregationError(
                    f"{source_path}:{line_number}: metric {metric!r} conflicts with "
                    "a generated identity column"
                )
            metrics[metric] = average

    if not metrics:
        raise AggregationError(f"{source_path}: contains no metric rows")
    return tuple(metrics), metrics


def is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def collect_summary(root: Path, output_dir: Path, source_name: str) -> UnifiedSummary:
    """Collect all inputs with one summary filename into unified rows."""
    source_paths = sorted(
        path
        for path in root.rglob(source_name)
        if path.is_file() and not is_within(path, output_dir)
    )
    if not source_paths:
        raise AggregationError(f"found no {source_name!r} files below {root}")

    rows: list[dict[str, str]] = []
    metric_columns: tuple[str, ...] | None = None
    seen_locations: set[tuple[str, str]] = set()

    for source_path in source_paths:
        identity = source_identity(root, source_path)
        location = (identity["model_path"], identity["prompt"])
        if location in seen_locations:
            raise AggregationError(
                f"duplicate {source_name} input for {location[0]}/results/{location[1]}"
            )
        seen_locations.add(location)

        current_columns, metrics = read_metric_table(source_path)
        if metric_columns is None:
            metric_columns = current_columns
        elif current_columns != metric_columns:
            raise AggregationError(
                f"{source_path}: metric names/order differ from other {source_name} "
                f"inputs; expected {list(metric_columns)!r}, found "
                f"{list(current_columns)!r}"
            )

        rows.append({**identity, **metrics})

    rows.sort(key=lambda row: (row["model_path"], row["prompt"]))
    assert metric_columns is not None
    return UnifiedSummary(source_name, metric_columns, tuple(rows))


def identity_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in IDENTITY_COLUMNS)


def validate_matching_sources(summaries: Sequence[UnifiedSummary]) -> None:
    """Require all four summary types to cover the same result locations."""
    reference = summaries[0]
    reference_keys = {identity_key(row) for row in reference.rows}
    for summary in summaries[1:]:
        current_keys = {identity_key(row) for row in summary.rows}
        missing = sorted(reference_keys - current_keys)
        extra = sorted(current_keys - reference_keys)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(
                    f"missing {len(missing)} location(s), including {missing[0]!r}"
                )
            if extra:
                details.append(
                    f"has {len(extra)} extra location(s), including {extra[0]!r}"
                )
            raise AggregationError(
                f"{summary.source_name} does not cover the same sources as "
                f"{reference.source_name}: {'; '.join(details)}"
            )


def write_summary(summary: UnifiedSummary, output_path: Path) -> None:
    """Atomically replace one unified CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=summary.columns)
            writer.writeheader()
            writer.writerows(summary.rows)
        os.replace(temporary_name, output_path)
    except OSError as error:
        raise AggregationError(f"could not write {output_path}: {error}") from error
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def read_unified(output_path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    try:
        handle = output_path.open(encoding="utf-8-sig", newline="")
    except OSError as error:
        raise AggregationError(f"could not read {output_path}: {error}") from error

    with handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        rows = tuple(dict(row) for row in reader)
    return columns, rows


def describe_row(row: dict[str, str]) -> str:
    return f"{row.get('model_path', '?')}/results/{row.get('prompt', '?')}"


def check_summary(summary: UnifiedSummary, output_path: Path) -> list[str]:
    """Return human-readable differences between expected and stored output."""
    if not output_path.is_file():
        return [f"output file does not exist: {output_path}"]

    actual_columns, actual_rows = read_unified(output_path)
    problems: list[str] = []
    if actual_columns != summary.columns:
        problems.append(
            f"column mismatch: expected {list(summary.columns)!r}, "
            f"found {list(actual_columns)!r}"
        )
        return problems

    expected_by_key = {identity_key(row): row for row in summary.rows}
    actual_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in actual_rows:
        key = identity_key(row)
        if key in actual_by_key:
            problems.append(f"duplicate output row: {describe_row(row)}")
        actual_by_key[key] = row

    missing_keys = sorted(expected_by_key.keys() - actual_by_key.keys())
    extra_keys = sorted(actual_by_key.keys() - expected_by_key.keys())
    for key in missing_keys:
        problems.append(f"missing output row: {describe_row(expected_by_key[key])}")
    for key in extra_keys:
        problems.append(f"stale/extra output row: {describe_row(actual_by_key[key])}")

    for key in sorted(expected_by_key.keys() & actual_by_key.keys()):
        expected = expected_by_key[key]
        actual = actual_by_key[key]
        differing_columns = [
            column for column in summary.columns if expected[column] != actual[column]
        ]
        if differing_columns:
            differences = ", ".join(
                f"{column}={actual[column]!r} (expected {expected[column]!r})"
                for column in differing_columns
            )
            problems.append(f"value mismatch at {describe_row(expected)}: {differences}")

    expected_order = [identity_key(row) for row in summary.rows]
    actual_order = [identity_key(row) for row in actual_rows]
    if not missing_keys and not extra_keys and expected_order != actual_order:
        problems.append("row order differs from model_path/prompt sort order")

    return problems


def print_problems(output_path: Path, problems: Iterable[str]) -> None:
    print(f"MISMATCH {output_path}", file=sys.stderr)
    problem_list = list(problems)
    for problem in problem_list[:20]:
        print(f"  - {problem}", file=sys.stderr)
    if len(problem_list) > 20:
        print(f"  - ... and {len(problem_list) - 20} more", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=script_dir.parent,
        help="result-tree root to scan (default: parent of this script's directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="directory for unified CSVs (default: this script's directory)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare existing outputs with the source tree; do not write files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()

    if not root.is_dir():
        print(f"error: result root is not a directory: {root}", file=sys.stderr)
        return 2

    try:
        summaries = tuple(
            collect_summary(root, output_dir, source_name)
            for source_name in SOURCE_FILES
        )
        validate_matching_sources(summaries)

        if args.check:
            mismatch_count = 0
            for summary in summaries:
                output_path = output_dir / summary.output_name
                problems = check_summary(summary, output_path)
                if problems:
                    mismatch_count += len(problems)
                    print_problems(output_path, problems)
                else:
                    print(
                        f"OK {output_path}: {len(summary.rows)} source files, "
                        f"{len(summary.rows)} exact output rows"
                    )
            return 1 if mismatch_count else 0

        for summary in summaries:
            output_path = output_dir / summary.output_name
            write_summary(summary, output_path)
            print(
                f"WROTE {output_path}: {len(summary.rows)} source files, "
                f"{len(summary.rows)} output rows"
            )
        return 0
    except (AggregationError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
