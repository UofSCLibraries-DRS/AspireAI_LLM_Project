import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from utils.config import ChatbotSpec, EvalDataConfig, ExperimentConfig, RagConfig
from utils.gaico import run_gaico


class RunGaicoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.out_dir = Path(self.temp_dir.name) / "Exp1"
        self.out_dir.mkdir()
        FakeExperiment.calls = []
        FakeTqdm.instances = []

    def test_writes_per_result_scores_and_zero_scores_for_failed_rows(self) -> None:
        self._write_results(
            [
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "chatbot_id": "bot-a",
                    "response": "Response one",
                    "error": "",
                },
                {
                    "scenario_id": "q-2",
                    "question": "Question two?",
                    "answer": "Answer two",
                    "chatbot_id": "bot-b",
                    "response": "",
                    "error": "failed",
                },
                {
                    "scenario_id": "q-3",
                    "question": "Question three?",
                    "answer": "Answer three",
                    "chatbot_id": "bot-c",
                    "response": "",
                    "error": "",
                },
                {
                    "scenario_id": "q-4",
                    "question": "Question four?",
                    "answer": "",
                    "chatbot_id": "bot-d",
                    "response": "Skipped",
                    "error": "",
                },
            ]
        )

        with patch("utils.gaico.Experiment", FakeExperiment):
            output_paths = run_gaico(self._config(["answer"]))

        self.assertEqual(output_paths, [self.out_dir / "gaico" / "answer.csv"])
        rows = self._read_scores("answer.csv")

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            list(rows[0].keys()),
            [
                "scenario_id",
                "question",
                "answer",
                "chatbot_id",
                "response",
                "error",
                "MetricA",
                "MetricB",
            ],
        )
        self.assertEqual(
            [row["scenario_id"] for row in rows],
            ["q-1", "q-2", "q-3"],
        )

        self.assertEqual(rows[0]["MetricA"], "0.5")
        self.assertEqual(rows[0]["MetricB"], "0.75")
        self.assertEqual(rows[1]["MetricA"], "0.0")
        self.assertEqual(rows[1]["MetricB"], "0.0")
        self.assertEqual(rows[2]["MetricA"], "0.0")
        self.assertEqual(rows[2]["MetricB"], "0.0")
        self.assertEqual(rows[0]["scenario_id"], "q-1")
        self.assertEqual(rows[0]["question"], "Question one?")

    def test_uses_default_metric_names_when_no_successful_rows_exist(self) -> None:
        self._write_results(
            [
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "chatbot_id": "bot-a",
                    "response": "",
                    "error": "failed",
                }
            ]
        )

        with patch("utils.gaico.Experiment", FakeExperiment):
            run_gaico(self._config(["answer"]))

        rows = self._read_scores("answer.csv")
        self.assertEqual(len(rows), 1)
        metric_names = [
            fieldname
            for fieldname in rows[0]
            if fieldname
            not in {
                "scenario_id",
                "question",
                "answer",
                "chatbot_id",
                "response",
                "error",
            }
        ]
        self.assertGreater(len(metric_names), 0)
        self.assertTrue(
            all(rows[0][metric_name] == "0.0" for metric_name in metric_names)
        )
        self.assertEqual(FakeExperiment.calls, [])

    def test_progress_tracks_scored_reference_groups(self) -> None:
        self._write_results(
            [
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "answer_extra": "Extra one",
                    "chatbot_id": "bot-a",
                    "response": "Response one",
                    "error": "",
                },
                {
                    "scenario_id": "q-2",
                    "question": "Question two?",
                    "answer": "Answer one",
                    "answer_extra": "Extra two",
                    "chatbot_id": "bot-b",
                    "response": "Response two",
                    "error": "",
                },
                {
                    "scenario_id": "q-3",
                    "question": "Question three?",
                    "answer": "Answer three",
                    "answer_extra": "Extra three",
                    "chatbot_id": "bot-c",
                    "response": "",
                    "error": "failed",
                },
            ],
            extra_fieldnames=["answer_extra"],
        )

        with (
            patch("utils.gaico.Experiment", FakeExperiment),
            patch("utils.gaico.tqdm", FakeTqdm),
        ):
            run_gaico(self._config(["answer", "answer_extra"]))

        self.assertEqual(len(FakeTqdm.instances), 1)
        progress = FakeTqdm.instances[0]
        self.assertEqual(progress.total, 3)
        self.assertEqual(progress.desc, "Running Gaico")
        self.assertEqual(progress.unit, "group")
        self.assertFalse(progress.disable)
        self.assertEqual(progress.updates, [1, 1, 1])

    def test_skips_progress_bar_when_no_successful_rows_exist(self) -> None:
        self._write_results(
            [
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "chatbot_id": "bot-a",
                    "response": "",
                    "error": "failed",
                }
            ]
        )

        with (
            patch("utils.gaico.Experiment", FakeExperiment),
            patch("utils.gaico.tqdm", FakeTqdm),
        ):
            run_gaico(self._config(["answer"]))

        self.assertEqual(len(FakeTqdm.instances), 1)
        progress = FakeTqdm.instances[0]
        self.assertEqual(progress.total, 0)
        self.assertTrue(progress.disable)
        self.assertEqual(progress.updates, [])
        self.assertEqual(FakeExperiment.calls, [])

    def test_suppresses_metric_stdout_and_stderr_noise(self) -> None:
        self._write_results(
            [
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "chatbot_id": "bot-a",
                    "response": "Response one",
                    "error": "",
                }
            ]
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("utils.gaico.Experiment", NoisyFakeExperiment),
            patch("utils.gaico.tqdm", FakeTqdm),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            run_gaico(self._config(["answer"]))

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_raises_when_ground_truth_column_is_missing(self) -> None:
        self._write_results(
            [
                {
                    "scenario_id": "q-1",
                    "question": "Question one?",
                    "answer": "Answer one",
                    "chatbot_id": "bot-a",
                    "response": "Response one",
                    "error": "",
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "ground truth column `expected`"):
            run_gaico(self._config(["expected"]))

    def _config(self, ground_truth_columns: list[str]) -> ExperimentConfig:
        return ExperimentConfig(
            out=str(self.out_dir),
            eval_data=EvalDataConfig(
                path="eval/scenarios.csv",
                question_column="question",
                ground_truth_columns=ground_truth_columns,
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

    def _write_results(
        self,
        rows: list[dict[str, str]],
        extra_fieldnames: list[str] | None = None,
    ) -> None:
        extra_fieldnames = extra_fieldnames or []
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
                    *extra_fieldnames,
                    "chatbot_id",
                    "response",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _read_scores(self, filename: str) -> list[dict[str, str]]:
        with (self.out_dir / "gaico" / filename).open(
            "r",
            encoding="utf-8",
            newline="",
        ) as scores_file:
            return list(csv.DictReader(scores_file))


class FakeExperiment:
    calls: list[dict] = []

    def __init__(self, llm_responses: dict[str, str], reference_answer: str) -> None:
        self.llm_responses = llm_responses
        self.reference_answer = reference_answer
        FakeExperiment.calls.append(
            {
                "llm_responses": llm_responses,
                "reference_answer": reference_answer,
            }
        )

    def compare(self, plot: bool) -> "FakeComparisonDataFrame":
        records = []
        for model_name in self.llm_responses:
            records.extend(
                [
                    {
                        "model_name": model_name,
                        "metric_name": "MetricA",
                        "score": 0.5,
                    },
                    {
                        "model_name": model_name,
                        "metric_name": "MetricB",
                        "score": 0.75,
                    },
                ]
            )
        return FakeComparisonDataFrame(records)


class NoisyFakeExperiment(FakeExperiment):
    def compare(self, plot: bool) -> "FakeComparisonDataFrame":
        from transformers.utils import logging as transformers_logging

        print("noisy stdout")
        print("noisy stderr", file=sys.stderr)
        transformers_logging.get_logger("transformers.utils.loading_report").warning(
            "noisy transformers warning"
        )
        return super().compare(plot=plot)


class FakeComparisonDataFrame:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def to_dict(self, orient: str) -> list[dict]:
        if orient != "records":
            raise ValueError(f"Unsupported orient: {orient}")
        return self.records


class FakeTqdm:
    instances: list["FakeTqdm"] = []

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str,
        disable: bool,
    ) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.updates: list[int] = []
        FakeTqdm.instances.append(self)

    def __enter__(self) -> "FakeTqdm":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self, n: int = 1) -> None:
        self.updates.append(n)


if __name__ == "__main__":
    unittest.main()
