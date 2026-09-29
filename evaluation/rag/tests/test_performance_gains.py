import csv
import tempfile
import unittest
from pathlib import Path

from utils.performance_gains import (
    analyze_directory,
    calculate_gains,
    load_score_data,
    parse_model_id,
    write_results,
)


class PerformanceGainsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.gaico_dir = Path(self.temp_dir.name) / "gaico"
        self.gaico_dir.mkdir()

    def test_calculates_each_one_factor_gain_from_matched_scenarios(self) -> None:
        score_path = self.gaico_dir / "answer_short.csv"
        rows = []
        for epochs in (2, 4):
            for mlp in (False, True):
                for rank in (8, 16, 32):
                    for retriever in (None, "naive"):
                        model_id = self._model_id(epochs, mlp, rank, retriever)
                        for scenario_number in (1, 2):
                            score = (
                                scenario_number
                                + (0.4 if epochs == 4 else 0.0)
                                + (0.2 if mlp else 0.0)
                                + {8: 0.0, 16: 0.1, 32: 0.3}[rank]
                                + (0.5 if retriever else 0.0)
                            )
                            rows.append(
                                {
                                    "scenario_id": f"q-{scenario_number}",
                                    "question": f"Question {scenario_number}?",
                                    "answer": f"Answer {scenario_number}",
                                    "chatbot_id": model_id,
                                    "input_prompt": "prompt",
                                    "response": "response",
                                    "error": "",
                                    "MetricA": score,
                                    "MetricB": score * 2,
                                }
                            )
        rows.append(
            {
                "scenario_id": "q-1",
                "question": "Question 1?",
                "answer": "Answer 1",
                "chatbot_id": "meta-llama/base",
                "input_prompt": "prompt",
                "response": "response",
                "error": "",
                "MetricA": 0.0,
                "MetricB": 0.0,
            }
        )
        self._write_scores(score_path, rows)

        score_data = load_score_data(score_path)
        results = calculate_gains(score_data)
        indexed = {
            (row.dimension, row.baseline, row.candidate, row.metric): row
            for row in results
        }

        expected = {
            ("epochs", "e2", "e4"): (0.4, 12, 24),
            ("mlp", "no_MLP", "MLP"): (0.2, 12, 24),
            ("rank", "r8", "r16"): (0.1, 8, 16),
            ("rank", "r16", "r32"): (0.2, 8, 16),
            ("rank", "r8", "r32"): (0.3, 8, 16),
            ("retrieval", "non-rag", "rag (naive)"): (0.5, 12, 24),
        }
        for comparison, (gain, pair_count, scenario_count) in expected.items():
            with self.subTest(comparison=comparison):
                result = indexed[(*comparison, "MetricA")]
                self.assertAlmostEqual(result.absolute_gain, gain)
                self.assertEqual(result.matched_model_pairs, pair_count)
                self.assertEqual(result.matched_scenarios, scenario_count)

        self.assertEqual(
            score_data.ignored_model_ids,
            frozenset({"meta-llama/base"}),
        )

    def test_averages_repeated_responses_before_comparing(self) -> None:
        score_path = self.gaico_dir / "answer_ideal.csv"
        rows = []
        for model_id, values in (
            ("M14.meta.e2.MLP.r8", (0.2, 0.4)),
            ("M14.meta.e4.MLP.r8", (0.7, 0.9)),
        ):
            for value in values:
                rows.append(
                    {
                        "scenario_id": "q-1",
                        "question": "Question?",
                        "answer": "Answer",
                        "chatbot_id": model_id,
                        "input_prompt": "prompt",
                        "response": "response",
                        "error": "",
                        "MetricA": value,
                    }
                )
        self._write_scores(score_path, rows)

        result = calculate_gains(load_score_data(score_path))[0]

        self.assertAlmostEqual(result.baseline_mean, 0.3)
        self.assertAlmostEqual(result.candidate_mean, 0.8)
        self.assertAlmostEqual(result.absolute_gain, 0.5)
        self.assertEqual(result.matched_scenarios, 1)

    def test_filters_metrics_and_writes_combined_csv(self) -> None:
        for source in ("answer_short.csv", "answer_ideal.csv"):
            self._write_scores(
                self.gaico_dir / source,
                [
                    self._simple_row("M14.meta.e2.MLP.r8", 0.25),
                    self._simple_row("M14.meta.e4.MLP.r8", 0.5),
                ],
            )

        results, ignored = analyze_directory(
            self.gaico_dir,
            requested_metrics=["MetricA"],
        )
        output_path = Path(self.temp_dir.name) / "gains.csv"
        write_results(results, output_path)

        with output_path.open(encoding="utf-8", newline="") as output_file:
            output_rows = list(csv.DictReader(output_file))
        self.assertEqual(len(output_rows), 2)
        self.assertEqual({row["metric"] for row in output_rows}, {"MetricA"})
        self.assertEqual(ignored, set())

    def test_rejects_unknown_metric(self) -> None:
        score_path = self.gaico_dir / "answer.csv"
        self._write_scores(
            score_path,
            [self._simple_row("M14.meta.e2.MLP.r8", 0.25)],
        )

        with self.assertRaisesRegex(ValueError, "Unknown metric"):
            load_score_data(score_path, requested_metrics=["MissingMetric"])

    def test_parses_rag_and_non_rag_model_ids(self) -> None:
        non_rag = parse_model_id("M14.meta.e2.no_MLP.r16")
        rag = parse_model_id("M14.meta.e2.no_MLP.r16 (naive)")

        self.assertIsNotNone(non_rag)
        self.assertIsNotNone(rag)
        self.assertIsNone(non_rag.retriever)
        self.assertEqual(rag.retriever, "naive")
        self.assertIsNone(parse_model_id("meta-llama/Meta-Llama-3.1-8B-Instruct"))

    @staticmethod
    def _model_id(
        epochs: int,
        mlp: bool,
        rank: int,
        retriever: str | None,
    ) -> str:
        model_id = f"M14.meta.e{epochs}.{'MLP' if mlp else 'no_MLP'}.r{rank}"
        return model_id if retriever is None else f"{model_id} ({retriever})"

    @staticmethod
    def _simple_row(model_id: str, metric_a: float) -> dict[str, str | float]:
        return {
            "scenario_id": "q-1",
            "question": "Question?",
            "answer": "Answer",
            "chatbot_id": model_id,
            "input_prompt": "prompt",
            "response": "response",
            "error": "",
            "MetricA": metric_a,
            "MetricB": metric_a * 2,
        }

    @staticmethod
    def _write_scores(
        score_path: Path,
        rows: list[dict[str, str | float]],
    ) -> None:
        with score_path.open("w", encoding="utf-8", newline="") as score_file:
            writer = csv.DictWriter(score_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
