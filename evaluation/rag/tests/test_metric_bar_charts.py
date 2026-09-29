import csv
import tempfile
import unittest
from pathlib import Path

from utils.metric_bar_charts import (
    load_configuration_scores,
    main,
)


class MetricBarChartsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.input_path = Path(self.temp_dir.name) / "model_rankings.csv"
        with self.input_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "subset",
                    "model_id",
                    "mean_score",
                    "mean_MetricA",
                    "mean_MetricB",
                ),
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "subset": "answer_ideal",
                        "model_id": "M14.meta.e2.MLP.r8",
                        "mean_score": 0.5,
                        "mean_MetricA": 0.8,
                        "mean_MetricB": 0.2,
                    },
                    {
                        "subset": "answer_ideal",
                        "model_id": "meta-llama/base (naive)",
                        "mean_score": 0.4,
                        "mean_MetricA": 0.6,
                        "mean_MetricB": 0.2,
                    },
                    {
                        "subset": "answer_short",
                        "model_id": "M14.meta.e2.MLP.r8",
                        "mean_score": 0.3,
                        "mean_MetricA": 0.4,
                        "mean_MetricB": 0.2,
                    },
                ]
            )

    def test_loads_only_per_metric_columns_and_calculates_config_mean(self) -> None:
        records, metric_names = load_configuration_scores(self.input_path)

        self.assertEqual(metric_names, ("MetricA", "MetricB"))
        self.assertEqual(len(records), 3)
        self.assertAlmostEqual(records[0].mean, 0.5)

    def test_cli_writes_metric_and_configuration_mean_charts(self) -> None:
        output_dir = Path(self.temp_dir.name) / "metric_bar_charts"

        exit_code = main(
            [
                str(self.input_path),
                "--output-dir",
                str(output_dir),
            ]
        )

        self.assertEqual(exit_code, 0)
        for subset in ("ideal", "short"):
            for filename in ("MetricA.png", "MetricB.png", "configuration_mean.png"):
                chart_path = output_dir / subset / filename
                self.assertGreater(chart_path.stat().st_size, 0)

    def test_rejects_input_without_metric_columns(self) -> None:
        empty_path = Path(self.temp_dir.name) / "empty.csv"
        with empty_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["subset", "model_id", "mean_score"])

        with self.assertRaisesRegex(ValueError, "no per-metric mean"):
            load_configuration_scores(empty_path)


if __name__ == "__main__":
    unittest.main()
