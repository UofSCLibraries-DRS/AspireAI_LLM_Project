#!/usr/bin/env python3
"""Report matched performance gains between M14 experiment settings.

The input is a directory of per-reference GAICO CSVs. Comparisons change one
setting at a time while keeping the model prefix, scenario, and every other
setting fixed. Repeated responses for the same model/scenario are averaged
before the matched differences are calculated.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_GAICO_DIR = Path("results/M14_naive_rag_greedy/gaico")
MODEL_ID_PATTERN = re.compile(
    r"^(?P<prefix>.+)\.e(?P<epochs>\d+)\."
    r"(?P<mlp>MLP|no_MLP)\.r(?P<rank>\d+)"
    r"(?: \((?P<retriever>[^()]+)\))?$"
)
OUTPUT_FIELDS = (
    "source",
    "dimension",
    "baseline",
    "candidate",
    "metric",
    "matched_model_pairs",
    "matched_scenarios",
    "baseline_mean",
    "candidate_mean",
    "absolute_gain",
    "relative_gain_percent",
)


@dataclass(frozen=True)
class ModelVariant:
    prefix: str
    epochs: int
    uses_mlp: bool
    rank: int
    retriever: str | None

    @property
    def label(self) -> str:
        mlp = "MLP" if self.uses_mlp else "no_MLP"
        label = f"{self.prefix}.e{self.epochs}.{mlp}.r{self.rank}"
        if self.retriever is not None:
            label += f" ({self.retriever})"
        return label


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count


ScenarioKey = tuple[str, ...]
ScenarioScores = dict[str, RunningAverage]
ScoresByModel = dict[ModelVariant, dict[ScenarioKey, ScenarioScores]]


@dataclass(frozen=True)
class ScoreData:
    source: str
    metrics: tuple[str, ...]
    scores: ScoresByModel
    ignored_model_ids: frozenset[str]


@dataclass(frozen=True)
class Comparison:
    dimension: str
    baseline: str
    candidate: str
    baseline_variant: ModelVariant
    candidate_variant: ModelVariant


@dataclass
class GainAccumulator:
    baseline_total: float = 0.0
    candidate_total: float = 0.0
    matched_scenarios: int = 0
    model_pairs: set[tuple[ModelVariant, ModelVariant]] = field(default_factory=set)

    def add(
        self,
        baseline_value: float,
        candidate_value: float,
        comparison: Comparison,
    ) -> None:
        self.baseline_total += baseline_value
        self.candidate_total += candidate_value
        self.matched_scenarios += 1
        self.model_pairs.add(
            (comparison.baseline_variant, comparison.candidate_variant)
        )


@dataclass(frozen=True)
class GainResult:
    source: str
    dimension: str
    baseline: str
    candidate: str
    metric: str
    matched_model_pairs: int
    matched_scenarios: int
    baseline_mean: float
    candidate_mean: float
    absolute_gain: float
    relative_gain_percent: float | None

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "source": self.source,
            "dimension": self.dimension,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "metric": self.metric,
            "matched_model_pairs": self.matched_model_pairs,
            "matched_scenarios": self.matched_scenarios,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "absolute_gain": self.absolute_gain,
            "relative_gain_percent": (
                ""
                if self.relative_gain_percent is None
                else self.relative_gain_percent
            ),
        }


def parse_model_id(model_id: str) -> ModelVariant | None:
    """Parse IDs such as ``M14.meta.e2.MLP.r8 (naive)``."""
    match = MODEL_ID_PATTERN.fullmatch(model_id.strip())
    if match is None:
        return None
    return ModelVariant(
        prefix=match.group("prefix"),
        epochs=int(match.group("epochs")),
        uses_mlp=match.group("mlp") == "MLP",
        rank=int(match.group("rank")),
        retriever=match.group("retriever"),
    )


def load_score_data(
    score_path: Path,
    requested_metrics: Sequence[str] | None = None,
) -> ScoreData:
    """Load one GAICO CSV and average repeated model/scenario scores."""
    csv.field_size_limit(sys.maxsize)
    with score_path.open("r", encoding="utf-8", newline="") as score_file:
        reader = csv.DictReader(score_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"GAICO CSV is empty: {score_path}")
        for required_field in ("chatbot_id", "error"):
            if required_field not in fieldnames:
                raise ValueError(
                    f"GAICO CSV must contain `{required_field}`: {score_path}"
                )

        chatbot_index = fieldnames.index("chatbot_id")
        error_index = fieldnames.index("error")
        if error_index <= chatbot_index:
            raise ValueError(
                f"GAICO CSV has an unexpected column order: {score_path}"
            )
        scenario_fields = fieldnames[:chatbot_index]
        available_metrics = tuple(fieldnames[error_index + 1 :])
        metrics = _select_metrics(
            available_metrics=available_metrics,
            requested_metrics=requested_metrics,
            score_path=score_path,
        )

        scores: ScoresByModel = {}
        ignored_model_ids: set[str] = set()
        for row in reader:
            model_id = row.get("chatbot_id", "")
            variant = parse_model_id(model_id)
            if variant is None:
                if model_id.strip():
                    ignored_model_ids.add(model_id)
                continue

            scenario_key = tuple(row.get(field, "") for field in scenario_fields)
            scenario_scores = scores.setdefault(variant, {}).setdefault(
                scenario_key,
                {},
            )
            for metric in metrics:
                value = _finite_float(row.get(metric, ""))
                if value is not None:
                    scenario_scores.setdefault(metric, RunningAverage()).add(value)

    if not scores:
        raise ValueError(f"GAICO CSV contains no recognized model IDs: {score_path}")
    return ScoreData(
        source=score_path.stem,
        metrics=metrics,
        scores=scores,
        ignored_model_ids=frozenset(ignored_model_ids),
    )


def calculate_gains(score_data: ScoreData) -> list[GainResult]:
    """Calculate one-factor gains using matched models and scenarios."""
    accumulators: dict[tuple[str, str, str, str], GainAccumulator] = {}
    for comparison in _build_comparisons(score_data.scores):
        baseline_scores = score_data.scores[comparison.baseline_variant]
        candidate_scores = score_data.scores[comparison.candidate_variant]
        for scenario_key in baseline_scores.keys() & candidate_scores.keys():
            baseline_metrics = baseline_scores[scenario_key]
            candidate_metrics = candidate_scores[scenario_key]
            for metric in score_data.metrics:
                if metric not in baseline_metrics or metric not in candidate_metrics:
                    continue
                key = (
                    comparison.dimension,
                    comparison.baseline,
                    comparison.candidate,
                    metric,
                )
                accumulators.setdefault(key, GainAccumulator()).add(
                    baseline_value=baseline_metrics[metric].mean,
                    candidate_value=candidate_metrics[metric].mean,
                    comparison=comparison,
                )

    results = []
    for (dimension, baseline, candidate, metric), accumulator in accumulators.items():
        baseline_mean = accumulator.baseline_total / accumulator.matched_scenarios
        candidate_mean = accumulator.candidate_total / accumulator.matched_scenarios
        absolute_gain = candidate_mean - baseline_mean
        relative_gain = (
            None
            if baseline_mean == 0.0
            else absolute_gain / abs(baseline_mean) * 100.0
        )
        results.append(
            GainResult(
                source=score_data.source,
                dimension=dimension,
                baseline=baseline,
                candidate=candidate,
                metric=metric,
                matched_model_pairs=len(accumulator.model_pairs),
                matched_scenarios=accumulator.matched_scenarios,
                baseline_mean=baseline_mean,
                candidate_mean=candidate_mean,
                absolute_gain=absolute_gain,
                relative_gain_percent=relative_gain,
            )
        )
    return sorted(results, key=_result_sort_key)


def analyze_directory(
    gaico_dir: Path,
    requested_metrics: Sequence[str] | None = None,
    excluded_paths: Iterable[Path] = (),
) -> tuple[list[GainResult], set[str]]:
    """Analyze every GAICO CSV in a directory."""
    if not gaico_dir.is_dir():
        raise ValueError(f"GAICO directory does not exist: {gaico_dir}")

    excluded = {path.resolve() for path in excluded_paths}
    score_paths = [
        path
        for path in sorted(gaico_dir.glob("*.csv"))
        if path.resolve() not in excluded
    ]
    if not score_paths:
        raise ValueError(f"No GAICO CSV files found in: {gaico_dir}")

    results: list[GainResult] = []
    ignored_model_ids: set[str] = set()
    for score_path in score_paths:
        score_data = load_score_data(score_path, requested_metrics)
        results.extend(calculate_gains(score_data))
        ignored_model_ids.update(score_data.ignored_model_ids)
    return results, ignored_model_ids


def write_results(results: Sequence[GainResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(result.as_csv_row() for result in results)


def print_results(results: Sequence[GainResult]) -> None:
    headers = (
        "source",
        "change",
        "metric",
        "before",
        "after",
        "gain",
        "gain %",
        "pairs",
        "n",
    )
    rows = [
        (
            result.source,
            f"{result.baseline} -> {result.candidate}",
            result.metric,
            f"{result.baseline_mean:.6f}",
            f"{result.candidate_mean:.6f}",
            f"{result.absolute_gain:+.6f}",
            (
                "n/a"
                if result.relative_gain_percent is None
                else f"{result.relative_gain_percent:+.2f}%"
            ),
            str(result.matched_model_pairs),
            str(result.matched_scenarios),
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(value.ljust(width) for value, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def _select_metrics(
    available_metrics: tuple[str, ...],
    requested_metrics: Sequence[str] | None,
    score_path: Path,
) -> tuple[str, ...]:
    if not available_metrics:
        raise ValueError(f"GAICO CSV contains no metric columns: {score_path}")
    if not requested_metrics:
        return available_metrics

    unknown = sorted(set(requested_metrics) - set(available_metrics))
    if unknown:
        raise ValueError(
            f"Unknown metric(s) in {score_path}: {', '.join(unknown)}. "
            f"Available metrics: {', '.join(available_metrics)}"
        )
    return tuple(dict.fromkeys(requested_metrics))


def _finite_float(value: str | None) -> float | None:
    try:
        number = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _build_comparisons(scores: ScoresByModel) -> list[Comparison]:
    variants = set(scores)
    comparisons: list[Comparison] = []
    for variant in sorted(variants, key=lambda item: item.label):
        if variant.epochs == 2:
            _append_comparison(
                comparisons,
                variants,
                variant,
                replace(variant, epochs=4),
                "epochs",
                "e2",
                "e4",
            )
        if not variant.uses_mlp:
            _append_comparison(
                comparisons,
                variants,
                variant,
                replace(variant, uses_mlp=True),
                "mlp",
                "no_MLP",
                "MLP",
            )
        for baseline_rank, candidate_rank in ((8, 16), (16, 32), (8, 32)):
            if variant.rank == baseline_rank:
                _append_comparison(
                    comparisons,
                    variants,
                    variant,
                    replace(variant, rank=candidate_rank),
                    "rank",
                    f"r{baseline_rank}",
                    f"r{candidate_rank}",
                )
        if variant.retriever is None:
            retrievers = sorted(
                candidate.retriever
                for candidate in variants
                if candidate.retriever is not None
                and _same_training_config(variant, candidate)
            )
            for retriever in retrievers:
                candidate = replace(variant, retriever=retriever)
                _append_comparison(
                    comparisons,
                    variants,
                    variant,
                    candidate,
                    "retrieval",
                    "non-rag",
                    f"rag ({retriever})",
                )
    return comparisons


def _same_training_config(first: ModelVariant, second: ModelVariant) -> bool:
    return (
        first.prefix,
        first.epochs,
        first.uses_mlp,
        first.rank,
    ) == (
        second.prefix,
        second.epochs,
        second.uses_mlp,
        second.rank,
    )


def _append_comparison(
    comparisons: list[Comparison],
    variants: set[ModelVariant],
    baseline: ModelVariant,
    candidate: ModelVariant,
    dimension: str,
    baseline_label: str,
    candidate_label: str,
) -> None:
    if candidate in variants:
        comparisons.append(
            Comparison(
                dimension=dimension,
                baseline=baseline_label,
                candidate=candidate_label,
                baseline_variant=baseline,
                candidate_variant=candidate,
            )
        )


def _result_sort_key(result: GainResult) -> tuple:
    dimension_order = {"epochs": 0, "mlp": 1, "rank": 2, "retrieval": 3}
    rank_order = {"r8": 0, "r16": 1, "r32": 2}
    return (
        result.source,
        dimension_order[result.dimension],
        rank_order.get(result.baseline, 0),
        rank_order.get(result.candidate, 0),
        result.candidate,
        result.metric,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gaico_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_GAICO_DIR,
        help=f"Directory containing GAICO CSVs (default: {DEFAULT_GAICO_DIR})",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help="Metric to include; repeat for multiple metrics (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the complete gain table to this CSV",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results, ignored_model_ids = analyze_directory(
            gaico_dir=args.gaico_dir,
            requested_metrics=args.metrics,
            excluded_paths=(() if args.output is None else (args.output,)),
        )
    except (OSError, csv.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if not results:
        print("error: no matched configuration comparisons were found", file=sys.stderr)
        return 1

    print_results(results)
    if args.output is not None:
        write_results(results, args.output)
        print(f"\nWrote {len(results)} rows to {args.output}")
    if ignored_model_ids:
        print(
            "\nIgnored model IDs without e#/MLP/r# configuration fields: "
            + ", ".join(sorted(ignored_model_ids)),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
