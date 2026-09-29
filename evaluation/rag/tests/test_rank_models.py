import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from utils.rank_models import main, rank_directory, write_rankings


class RankModelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.gaico_dir = Path(self.temp_dir.name) / "gaico"
        self.gaico_dir.mkdir()
        self._write_scores(
            "answer_short.csv",
            [
                ("q1", "history", "alpha", "", 0.9, 0.8),
                ("q2", "history", "alpha", "", 0.7, 0.6),
                ("q1", "history", "beta", "", 0.6, 0.5),
                ("q1", "history", "gamma", "", 1.0, ""),
                ("q3", "politics", "alpha", "", 0.1, 0.1),
                ("q3", "politics", "beta", "", 0.9, 0.9),
            ],
        )
        self._write_scores(
            "answer_ideal.csv",
            [
                ("q1", "history", "alpha", "", 0.4, 0.9),
                ("q1", "history", "beta", "", 0.8, 0.5),
                ("q3", "politics", "alpha", "", 0.2, 0.2),
                ("q3", "politics", "beta", "", 0.8, 0.8),
            ],
        )

    def test_ranks_each_score_file_on_matched_scenarios(self) -> None:
        analysis = rank_directory(self.gaico_dir)
        ideal = [row for row in analysis.rankings if row.subset == "answer_ideal"]
        short = [row for row in analysis.rankings if row.subset == "answer_short"]

        self.assertEqual([row.rank for row in ideal], [1, 2])
        self.assertEqual([row.model_id for row in ideal], ["beta", "alpha"])
        self.assertEqual(ideal[0].sources, 1)
        self.assertEqual(ideal[0].components, 2)
        self.assertEqual(ideal[0].matched_scenarios, 2)

        self.assertEqual([row.rank for row in short], [1, 2])
        self.assertEqual([row.model_id for row in short], ["beta", "alpha"])
        self.assertAlmostEqual(short[0].ranking_score, 1.0)
        self.assertAlmostEqual(short[0].average_component_rank, 1.0)
        self.assertAlmostEqual(short[0].mean_score, 0.725)
        self.assertAlmostEqual(short[0].metric_means["MetricA"], 0.75)
        self.assertEqual(short[0].matched_scenarios, 2)
        self.assertEqual(analysis.ignored_incomplete_models, frozenset({"gamma"}))

    def test_supports_lower_is_better_metrics_per_score_file(self) -> None:
        analysis = rank_directory(
            self.gaico_dir,
            requested_metrics=["MetricB"],
            lower_is_better=["MetricB"],
        )
        winners = {
            row.subset: row.model_id for row in analysis.rankings if row.rank == 1
        }

        self.assertEqual(
            winners,
            {"answer_ideal": "alpha", "answer_short": "alpha"},
        )

    def test_rejects_unknown_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown metric"):
            rank_directory(self.gaico_dir, requested_metrics=["Missing"])
        with self.assertRaisesRegex(ValueError, "lower-is-better"):
            rank_directory(self.gaico_dir, lower_is_better=["Missing"])

    def test_writes_subset_csv_and_cli_limits_each_subset(self) -> None:
        output_path = self.gaico_dir / "model_rankings.csv"
        stdout = StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    str(self.gaico_dir),
                    "--metric",
                    "MetricA",
                    "--top",
                    "1",
                    "--output",
                    str(output_path),
                ]
            )

        self.assertEqual(exit_code, 0)
        printed = stdout.getvalue()
        self.assertIn("subset: answer_ideal", printed)
        self.assertIn("subset: answer_short", printed)
        with output_path.open("r", encoding="utf-8", newline="") as output_file:
            rows = list(csv.DictReader(output_file))
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [(row["subset"], row["rank"]) for row in rows],
            [
                ("answer_ideal", "1"),
                ("answer_ideal", "2"),
                ("answer_short", "1"),
                ("answer_short", "2"),
                ("answer_short", "3"),
            ],
        )
        self.assertIn("mean_MetricA", rows[0])

    def test_write_rankings_creates_parent_directory(self) -> None:
        analysis = rank_directory(self.gaico_dir, requested_metrics=["MetricB"])
        output_path = Path(self.temp_dir.name) / "nested" / "rankings.csv"

        write_rankings(analysis, output_path)

        self.assertTrue(output_path.is_file())

    def _write_scores(
        self,
        filename: str,
        rows: list[tuple[str, str, str, str, float, float | str]],
    ) -> None:
        with (self.gaico_dir / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as score_file:
            writer = csv.writer(score_file)
            writer.writerow(
                [
                    "question",
                    "subset",
                    "chatbot_id",
                    "error",
                    "MetricA",
                    "MetricB",
                ]
            )
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
