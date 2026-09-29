#!/usr/bin/env python3
"""Rank evaluated models from existing GAICO score CSVs.

Rankings are calculated independently for each input CSV (for example,
``answer_ideal`` or ``answer_short_agg``). Within one of those score subsets,
each model is ranked for every available metric.
Those ranks are converted to a 0-1 percentile score and averaged with equal
weight, preventing metrics with wider numeric ranges from dominating the final
ranking. Models are compared only on scenarios available for every model, and
repeated responses are averaged within each scenario first.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_GAICO_DIR = Path("results/M14_greedy/gaico")
BASE_OUTPUT_FIELDS = (
    "subset",
    "rank",
    "model_id",
    "ranking_score",
    "average_component_rank",
    "mean_score",
    "sources",
    "components",
    "matched_scenarios",
)


@dataclass
class RunningMean:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count


ScenarioKey = tuple[str, ...]

@dataclass(frozen=True)
class SourceScores:
    source: str
    metrics: tuple[str, ...]
    scores: dict[str, dict[ScenarioKey, dict[str, RunningMean]]]


@dataclass(frozen=True)
class ModelRanking:
    subset: str
    rank: int
    model_id: str
    ranking_score: float
    average_component_rank: float
    mean_score: float
    sources: int
    components: int
    matched_scenarios: int
    metric_means: dict[str, float]

    def as_csv_row(self, metrics: Sequence[str]) -> dict[str, str | int | float]:
        row: dict[str, str | int | float] = {
            "subset": self.subset,
            "rank": self.rank,
            "model_id": self.model_id,
            "ranking_score": self.ranking_score,
            "average_component_rank": self.average_component_rank,
            "mean_score": self.mean_score,
            "sources": self.sources,
            "components": self.components,
            "matched_scenarios": self.matched_scenarios,
        }
        for metric in metrics:
            row[f"mean_{metric}"] = self.metric_means.get(metric, "")
        return row


@dataclass(frozen=True)
class RankingAnalysis:
    rankings: tuple[ModelRanking, ...]
    metrics: tuple[str, ...]
    ignored_incomplete_models: frozenset[str] = field(default_factory=frozenset)


def load_source_scores(
    score_path: Path,
    requested_metrics: Sequence[str] | None = None,
) -> SourceScores:
    """Load one GAICO CSV, averaging repeats within each model/scenario."""
    csv.field_size_limit(sys.maxsize)
    with score_path.open("r", encoding="utf-8", newline="") as score_file:
        reader = csv.DictReader(score_file)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"GAICO CSV is empty: {score_path}")
        for required_field in ("subset", "chatbot_id", "error"):
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
            available_metrics,
            requested_metrics,
            score_path,
        )

        scores: dict[str, dict[ScenarioKey, dict[str, RunningMean]]] = {}
        for row in reader:
            model_id = row.get("chatbot_id", "").strip()
            if not model_id:
                continue

            finite_values = {
                metric: value
                for metric in metrics
                if (value := _finite_float(row.get(metric))) is not None
            }
            if not finite_values:
                continue

            scenario_key = tuple(row.get(field, "") for field in scenario_fields)
            scenario_scores = scores.setdefault(model_id, {}).setdefault(
                scenario_key,
                {},
            )
            for metric, value in finite_values.items():
                scenario_scores.setdefault(metric, RunningMean()).add(value)

    if not scores:
        raise ValueError(f"GAICO CSV contains no finite model scores: {score_path}")
    return SourceScores(
        source=score_path.stem,
        metrics=metrics,
        scores=scores,
    )


def rank_directory(
    gaico_dir: Path,
    requested_metrics: Sequence[str] | None = None,
    lower_is_better: Iterable[str] = (),
    excluded_paths: Iterable[Path] = (),
) -> RankingAnalysis:
    """Rank models using every selected metric in every GAICO CSV."""
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

    sources = [
        load_source_scores(path, requested_metrics=requested_metrics)
        for path in score_paths
    ]
    metrics = tuple(
        dict.fromkeys(metric for source in sources for metric in source.metrics)
    )
    lower_metrics = set(lower_is_better)
    unknown_lower_metrics = sorted(lower_metrics - set(metrics))
    if unknown_lower_metrics:
        raise ValueError(
            "Unknown --lower-is-better metric(s): "
            + ", ".join(unknown_lower_metrics)
            + ". Selected metrics: "
            + ", ".join(metrics)
        )

    rankings: list[ModelRanking] = []
    ignored_models: set[str] = set()
    for source in sources:
        subset_rankings, subset_ignored = _rank_subset(
            subset=source.source,
            sources=[source],
            metrics=metrics,
            lower_metrics=lower_metrics,
        )
        rankings.extend(subset_rankings)
        ignored_models.update(subset_ignored)

    return RankingAnalysis(
        rankings=tuple(rankings),
        metrics=metrics,
        ignored_incomplete_models=frozenset(ignored_models),
    )


def _rank_subset(
    *,
    subset: str,
    sources: Sequence[SourceScores],
    metrics: Sequence[str],
    lower_metrics: set[str],
) -> tuple[list[ModelRanking], set[str]]:
    components: list[tuple[int, SourceScores, str, set[str]]] = []
    all_models: set[str] = set()
    for source_index, source in enumerate(sources):
        models_in_subset = set(source.scores)
        all_models.update(models_in_subset)
        for metric in source.metrics:
            models = {
                model_id
                for model_id in models_in_subset
                if any(
                    metric in scenario_scores
                    for scenario_scores in source.scores[model_id].values()
                )
            }
            components.append((source_index, source, metric, models))

    complete_models = set.intersection(
        *(models for _, _, _, models in components)
    )
    if not complete_models:
        raise ValueError(
            f"No model has complete selected scores for subset {subset!r}"
        )
    ignored_models = all_models - complete_models

    matched_scenarios_by_source: dict[int, set[ScenarioKey]] = {}
    for source_index, source in enumerate(sources):
        scenarios_by_model = []
        for model_id in complete_models:
            scenarios_by_model.append(
                {
                    scenario_key
                    for scenario_key, scenario_scores in source.scores[
                        model_id
                    ].items()
                    if all(metric in scenario_scores for metric in source.metrics)
                }
            )
        matched_scenarios = set.intersection(*scenarios_by_model)
        if not matched_scenarios:
            raise ValueError(
                f"No matched scenarios remain for subset {subset!r} "
                f"in {source.source}"
            )
        matched_scenarios_by_source[source_index] = matched_scenarios
    matched_scenario_count = sum(
        len(scenarios) for scenarios in matched_scenarios_by_source.values()
    )

    ranks_by_model: dict[str, list[float]] = {
        model_id: [] for model_id in complete_models
    }
    percentiles_by_model: dict[str, list[float]] = {
        model_id: [] for model_id in complete_models
    }
    raw_scores_by_model: dict[str, list[float]] = {
        model_id: [] for model_id in complete_models
    }
    metric_scores_by_model: dict[str, dict[str, list[float]]] = {
        model_id: {metric: [] for metric in metrics}
        for model_id in complete_models
    }

    for source_index, source, metric, _ in components:
        matched_scenarios = matched_scenarios_by_source[source_index]
        values = {
            model_id: sum(
                source.scores[model_id][scenario_key][metric].mean
                for scenario_key in matched_scenarios
            )
            / len(matched_scenarios)
            for model_id in complete_models
        }
        component_ranks = _average_tie_ranks(
            values,
            lower_is_better=metric in lower_metrics,
        )
        denominator = len(complete_models) - 1
        for model_id, value in values.items():
            component_rank = component_ranks[model_id]
            percentile = (
                1.0
                if denominator == 0
                else 1.0 - (component_rank - 1.0) / denominator
            )
            ranks_by_model[model_id].append(component_rank)
            percentiles_by_model[model_id].append(percentile)
            raw_scores_by_model[model_id].append(value)
            metric_scores_by_model[model_id][metric].append(value)

    unordered = []
    for model_id in complete_models:
        component_ranks = ranks_by_model[model_id]
        metric_means = {
            metric: sum(values) / len(values)
            for metric, values in metric_scores_by_model[model_id].items()
            if values
        }
        unordered.append(
            ModelRanking(
                subset=subset,
                rank=0,
                model_id=model_id,
                ranking_score=sum(percentiles_by_model[model_id])
                / len(percentiles_by_model[model_id]),
                average_component_rank=sum(component_ranks) / len(component_ranks),
                mean_score=sum(raw_scores_by_model[model_id])
                / len(raw_scores_by_model[model_id]),
                sources=len(sources),
                components=len(component_ranks),
                matched_scenarios=matched_scenario_count,
                metric_means=metric_means,
            )
        )

    ordered = sorted(
        unordered,
        key=lambda item: (
            -item.ranking_score,
            item.average_component_rank,
            item.model_id,
        ),
    )
    rankings = [
        ModelRanking(
            subset=result.subset,
            rank=index,
            model_id=result.model_id,
            ranking_score=result.ranking_score,
            average_component_rank=result.average_component_rank,
            mean_score=result.mean_score,
            sources=result.sources,
            components=result.components,
            matched_scenarios=result.matched_scenarios,
            metric_means=result.metric_means,
        )
        for index, result in enumerate(ordered, start=1)
    ]
    return rankings, ignored_models


def write_rankings(analysis: RankingAnalysis, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = [f"mean_{metric}" for metric in analysis.metrics]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[*BASE_OUTPUT_FIELDS, *metric_fields],
        )
        writer.writeheader()
        writer.writerows(
            result.as_csv_row(analysis.metrics) for result in analysis.rankings
        )


def print_rankings(analysis: RankingAnalysis, top: int | None = None) -> None:
    rankings_by_subset: dict[str, list[ModelRanking]] = {}
    for result in analysis.rankings:
        rankings_by_subset.setdefault(result.subset, []).append(result)

    headers = ("rank", "model", "ranking", "mean", "avg rank", "scenarios")
    for subset_index, (subset, subset_rankings) in enumerate(
        rankings_by_subset.items()
    ):
        if subset_index:
            print()
        print(f"subset: {subset or '(blank)'}")
        results = subset_rankings if top is None else subset_rankings[:top]
        rows = [
            (
                str(result.rank),
                result.model_id,
                f"{result.ranking_score:.6f}",
                f"{result.mean_score:.6f}",
                f"{result.average_component_rank:.3f}",
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
def _average_tie_ranks(
    values: dict[str, float],
    *,
    lower_is_better: bool,
) -> dict[str, float]:
    ordered = sorted(
        values.items(),
        key=lambda item: (
            item[1] if lower_is_better else -item[1],
            item[0],
        ),
    )
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1],
            ordered[start][1],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for model_id, _ in ordered[start:end]:
            ranks[model_id] = average_rank
        start = end
    return ranks


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


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


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
        "--lower-is-better",
        action="append",
        default=[],
        metavar="METRIC",
        help="Treat a selected metric as lower-is-better; repeat as needed",
    )
    parser.add_argument(
        "--top",
        type=_positive_int,
        help="Only print the top N models (the output CSV still contains all models)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the complete ranking and per-metric means to this CSV",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        analysis = rank_directory(
            gaico_dir=args.gaico_dir,
            requested_metrics=args.metrics,
            lower_is_better=args.lower_is_better,
            excluded_paths=(() if args.output is None else (args.output,)),
        )
    except (OSError, csv.Error, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print_rankings(analysis, top=args.top)
    if args.output is not None:
        write_rankings(analysis, args.output)
        print(f"\nWrote {len(analysis.rankings)} model rankings to {args.output}")
    if analysis.ignored_incomplete_models:
        print(
            "\nIgnored models without complete selected scores: "
            + ", ".join(sorted(analysis.ignored_incomplete_models)),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
