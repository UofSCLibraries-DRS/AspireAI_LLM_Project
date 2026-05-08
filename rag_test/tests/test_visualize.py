import csv
import tempfile
import unittest
from pathlib import Path

from utils.config import ChatbotSpec, EvalDataConfig, ExperimentConfig, RagConfig
from utils.visualize import run_visualize


class RunVisualizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.out_dir = Path(self.temp_dir.name) / "Exp1"
        self.gaico_dir = self.out_dir / "gaico"
        self.gaico_dir.mkdir(parents=True)

    def test_writes_radar_chart_for_each_gaico_csv(self) -> None:
        self._write_results()
        self._write_gaico_csv(
            "answer.csv",
            [
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "chatbot_id": "bot-a",
                    "response": "Response one",
                    "error": "",
                    "MetricA": "0.25",
                    "MetricB": "0.75",
                },
                {
                    "scenario_id": "q-2",
                    "question": "Question two?",
                    "answer": "Answer two",
                    "chatbot_id": "bot-a",
                    "response": "Response two",
                    "error": "",
                    "MetricA": "0.75",
                    "MetricB": "0.25",
                },
                {
                    "scenario_id": "q-3",
                    "question": "Question three?",
                    "answer": "Answer three",
                    "chatbot_id": "bot-b",
                    "response": "Response three",
                    "error": "",
                    "MetricA": "1.0",
                    "MetricB": "0.5",
                },
            ],
        )

        output_paths = run_visualize(self._config())

        self.assertEqual(output_paths, [self.out_dir / "figures" / "answer_radar.png"])
        self.assertTrue(output_paths[0].exists())
        self.assertEqual(output_paths[0].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_raises_when_gaico_csv_has_no_metric_columns(self) -> None:
        self._write_results()
        self._write_gaico_csv(
            "answer.csv",
            [
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "chatbot_id": "bot-a",
                    "response": "Response one",
                    "error": "",
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "no metric columns"):
            run_visualize(self._config())

    def test_raises_when_gaico_folder_is_missing(self) -> None:
        self.gaico_dir.rmdir()
        self._write_results()

        with self.assertRaisesRegex(FileNotFoundError, "Gaico output folder"):
            run_visualize(self._config())

    def _config(self) -> ExperimentConfig:
        return ExperimentConfig(
            out=str(self.out_dir),
            eval_data=EvalDataConfig(
                path="eval/scenarios.csv",
                question_column="question",
                ground_truth_columns=["answer"],
            ),
            data="raw/combined.csv",
            rag_config=RagConfig(
                embedding_model="embedding-model",
                top_k=3,
            ),
            chatbots=[
                ChatbotSpec(
                    id="bot-a",
                    backend="DummyChatbot",
                    config={},
                )
            ],
        )

    def _write_results(self) -> None:
        self.out_dir.mkdir(exist_ok=True)
        with (self.out_dir / "results.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as results_file:
            writer = csv.DictWriter(
                results_file,
                fieldnames=[
                    "scenario_id",
                    "question",
                    "answer",
                    "chatbot_id",
                    "response",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "chatbot_id": "bot-a",
                    "response": "Response one",
                    "error": "",
                }
            )

    def _write_gaico_csv(self, filename: str, rows: list[dict[str, str]]) -> None:
        with (self.gaico_dir / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as gaico_file:
            writer = csv.DictWriter(gaico_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
