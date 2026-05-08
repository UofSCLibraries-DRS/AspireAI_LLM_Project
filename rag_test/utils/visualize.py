import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import ExperimentConfig
from utils.gaico import GAICO_OUTPUT_DIRNAME, RESULT_FILENAME


FIGURES_OUTPUT_DIRNAME = "figures"
MODEL_COLUMN = "chatbot_id"


def run_visualize(exp_config: ExperimentConfig) -> list[Path]:
    out_dir = Path(exp_config.out)
    result_path = out_dir / RESULT_FILENAME
    gaico_dir = out_dir / GAICO_OUTPUT_DIRNAME
    figures_dir = out_dir / FIGURES_OUTPUT_DIRNAME

    if not result_path.exists():
        raise FileNotFoundError(f"Experiment results not found: {result_path}")
    if not gaico_dir.exists():
        raise FileNotFoundError(f"Gaico output folder not found: {gaico_dir}")

    source_fieldnames = _load_result_fieldnames(result_path)
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for gaico_csv_path in sorted(gaico_dir.glob("*.csv")):
        output_path = figures_dir / f"{gaico_csv_path.stem}_radar.png"
        _write_radar_chart(
            gaico_csv_path=gaico_csv_path,
            source_fieldnames=source_fieldnames,
            output_path=output_path,
        )
        output_paths.append(output_path)

    return output_paths


def _load_result_fieldnames(result_path: Path) -> list[str]:
    with result_path.open("r", encoding="utf-8", newline="") as result_file:
        reader = csv.DictReader(result_file)
        if reader.fieldnames is None:
            raise ValueError(f"Results CSV is empty: {result_path}")
        return reader.fieldnames


def _write_radar_chart(
    gaico_csv_path: Path,
    source_fieldnames: list[str],
    output_path: Path,
) -> None:
    rows, fieldnames = _load_gaico_rows(gaico_csv_path)
    metric_names = [
        fieldname for fieldname in fieldnames if fieldname not in source_fieldnames
    ]
    if not metric_names:
        raise ValueError(f"Gaico CSV has no metric columns: {gaico_csv_path}")
    if MODEL_COLUMN not in fieldnames:
        raise ValueError(f"Gaico CSV must contain `{MODEL_COLUMN}`: {gaico_csv_path}")

    model_scores = _average_metrics_by_model(rows, metric_names)
    if not model_scores:
        raise ValueError(f"Gaico CSV has no plottable model scores: {gaico_csv_path}")

    _plot_radar_chart(
        metric_names=metric_names,
        model_scores=model_scores,
        title=gaico_csv_path.stem,
        output_path=output_path,
    )


def _load_gaico_rows(gaico_csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with gaico_csv_path.open("r", encoding="utf-8", newline="") as gaico_file:
        reader = csv.DictReader(gaico_file)
        if reader.fieldnames is None:
            raise ValueError(f"Gaico CSV is empty: {gaico_csv_path}")
        return list(reader), reader.fieldnames


def _average_metrics_by_model(
    rows: list[dict[str, str]],
    metric_names: list[str],
) -> dict[str, list[float]]:
    totals: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}

    for row in rows:
        model = row.get(MODEL_COLUMN, "").strip()
        if not model:
            continue

        totals.setdefault(model, {metric_name: 0.0 for metric_name in metric_names})
        counts.setdefault(model, {metric_name: 0 for metric_name in metric_names})

        for metric_name in metric_names:
            try:
                score = float(row.get(metric_name, ""))
            except ValueError:
                continue
            if not math.isfinite(score):
                continue

            totals[model][metric_name] += score
            counts[model][metric_name] += 1

    model_scores = {}
    for model, metric_totals in totals.items():
        model_scores[model] = [
            metric_totals[metric_name] / counts[model][metric_name]
            if counts[model][metric_name]
            else 0.0
            for metric_name in metric_names
        ]
    return model_scores


def _plot_radar_chart(
    metric_names: list[str],
    model_scores: dict[str, list[float]],
    title: str,
    output_path: Path,
) -> None:
    angles = [
        index / float(len(metric_names)) * 2 * math.pi
        for index in range(len(metric_names))
    ]
    closed_angles = [*angles, angles[0]]

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})
    for model, scores in sorted(model_scores.items()):
        closed_scores = [*scores, scores[0]]
        ax.plot(closed_angles, closed_scores, linewidth=2, label=model)
        ax.fill(closed_angles, closed_scores, alpha=0.12)

    ax.set_xticks(angles)
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylim(0, _score_axis_max(model_scores))
    ax.set_title(f"Gaico Metrics: {title}", pad=24)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _score_axis_max(model_scores: dict[str, list[float]]) -> float:
    max_score = max(score for scores in model_scores.values() for score in scores)
    return max(1.0, max_score)
