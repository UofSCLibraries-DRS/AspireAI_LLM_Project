import csv
import tempfile
import unittest
from pathlib import Path

from utils.group_result_matrices import (
    build_average_matrix,
    build_max_matrix,
    classify_model,
    load_model_values,
    main,
)


class GroupResultMatricesTests(unittest.TestCase):
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
                    "ranking_score",
                    "mean_score",
                ),
            )
            writer.writeheader()
            writer.writerows(
                [
                    self._row("answer_ideal", "M14.meta.e2.MLP.r8", 0.8, 0.4),
                    self._row(
                        "answer_ideal",
                        "M14.meta.e4.no_MLP.r16",
                        0.6,
                        0.3,
                    ),
                    self._row("answer_ideal", "meta-llama/base", 0.5, 0.2),
                    self._row(
                        "answer_ideal",
                        "M14.meta.e2.MLP.r8 (naive)",
                        0.9,
                        0.5,
                    ),
                    self._row(
                        "answer_ideal",
                        "meta-llama/base (naive)",
                        0.7,
                        0.35,
                    ),
                    self._row("answer_short", "M14.meta.e2.MLP.r8", 0.4, 0.1),
                ]
            )

    def test_classifies_rag_and_fine_tuning(self) -> None:
        self.assertEqual(
            classify_model("M14.meta.e2.no_MLP.r16 (hybrid-rrf)"),
            ("hybrid-rrf", "fine_tuned"),
        )
        self.assertEqual(
            classify_model("meta-llama/Meta-Llama-3.1-8B-Instruct"),
            ("none", "not_fine_tuned"),
        )

    def test_averages_models_within_each_cell(self) -> None:
        averages = build_average_matrix(load_model_values(self.input_path))
        ideal_none = next(
            row
            for row in averages
            if row["subset"] == "answer_ideal" and row["rag_type"] == "none"
        )

        self.assertAlmostEqual(ideal_none["fine_tuned"], 0.7)
        self.assertAlmostEqual(ideal_none["not_fine_tuned"], 0.5)

    def test_max_matrix_reports_winning_model_and_score(self) -> None:
        maxima = build_max_matrix(load_model_values(self.input_path))
        ideal_none = next(
            row
            for row in maxima
            if row["subset"] == "answer_ideal" and row["rag_type"] == "none"
        )

        self.assertAlmostEqual(ideal_none["fine_tuned"], 0.8)
        self.assertEqual(
            ideal_none["fine_tuned_model"],
            "M14.meta.e2.MLP.r8",
        )
        self.assertAlmostEqual(ideal_none["not_fine_tuned"], 0.5)
        self.assertEqual(ideal_none["not_fine_tuned_model"], "meta-llama/base")

    def test_supports_another_numeric_column(self) -> None:
        model_values = load_model_values(self.input_path, "mean_score")
        averages = build_average_matrix(model_values)
        ideal_none = next(
            row
            for row in averages
            if row["subset"] == "answer_ideal" and row["rag_type"] == "none"
        )
        self.assertAlmostEqual(ideal_none["fine_tuned"], 0.35)

    def test_cli_writes_matrices_and_color_coded_figures(self) -> None:
        average_path = Path(self.temp_dir.name) / "out" / "average.csv"
        max_path = Path(self.temp_dir.name) / "out" / "max.csv"
        figures_dir = Path(self.temp_dir.name) / "figures" / "rag_vs_fine-tuning"

        exit_code = main(
            [
                str(self.input_path),
                "--value-column",
                "mean_score",
                "--average-output",
                str(average_path),
                "--max-output",
                str(max_path),
                "--figures-dir",
                str(figures_dir),
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(average_path.is_file())
        self.assertTrue(max_path.is_file())
        for subset in ("ideal", "short"):
            self.assertGreater((figures_dir / subset / "average.png").stat().st_size, 0)
            self.assertGreater((figures_dir / subset / "max.png").stat().st_size, 0)

        with max_path.open("r", encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.assertEqual(
            list(rows[0]),
            [
                "subset",
                "rag_type",
                "fine_tuned",
                "fine_tuned_model",
                "not_fine_tuned",
                "not_fine_tuned_model",
            ],
        )

    def test_rejects_missing_value_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing column"):
            load_model_values(self.input_path, "missing")

    @staticmethod
    def _row(
        subset: str,
        model_id: str,
        ranking_score: float,
        mean_score: float,
    ) -> dict[str, str | float]:
        return {
            "subset": subset,
            "model_id": model_id,
            "ranking_score": ranking_score,
            "mean_score": mean_score,
        }


if __name__ == "__main__":
    unittest.main()
