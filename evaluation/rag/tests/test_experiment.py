import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from utils.config import (
    ChatbotSpec,
    EvalDataConfig,
    ExperimentConfig,
    RagConfig,
    RetrieverSpec,
)
from utils.experiment import run_experiment


class RunExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.out_dir = Path(self.temp_dir.name) / "Exp1"
        FakeTqdm.instances = []

    def test_repeats_generation_without_rebuilding_prompts(self) -> None:
        chatbot = FakeChatbot()
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]
        prompts = ["Prompt one", "Prompt two"]

        with (
            patch("utils.experiment.create_retriever", return_value=object()),
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch("utils.experiment.build_rag_prompts", return_value=prompts) as build,
            patch("utils.experiment.create_chatbot", return_value=chatbot),
        ):
            result_path = run_experiment(self._config(), k=3)

        self.assertEqual(
            chatbot.prompts,
            [
                "Prompt one",
                "Prompt one",
                "Prompt one",
                "Prompt two",
                "Prompt two",
                "Prompt two",
            ],
        )
        build.assert_called_once()

        rows = self._read_results(result_path)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            list(rows[0].keys()),
            [
                "scenario_id",
                "question",
                "chatbot_id",
                "input_prompt",
                "response",
                "error",
            ],
        )
        self.assertEqual(
            [row["scenario_id"] for row in rows],
            ["q-1", "q-1", "q-1", "q-2", "q-2", "q-2"],
        )
        self.assertEqual({row["chatbot_id"] for row in rows}, {"bot-a (fake)"})
        self.assertEqual(
            [row["input_prompt"] for row in rows],
            [
                "Prompt one",
                "Prompt one",
                "Prompt one",
                "Prompt two",
                "Prompt two",
                "Prompt two",
            ],
        )

    def test_uses_configured_batch_size_and_preserves_row_order(self) -> None:
        chatbot = FakeChatbot()
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]
        prompts = ["Prompt one", "Prompt two"]

        with (
            patch("utils.experiment.create_retriever", return_value=object()),
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch("utils.experiment.build_rag_prompts", return_value=prompts),
            patch("utils.experiment.create_chatbot", return_value=chatbot),
        ):
            result_path = run_experiment(self._config(batch_size=2), k=2)

        self.assertEqual(
            chatbot.batch_prompts,
            [["Prompt one", "Prompt one"], ["Prompt two", "Prompt two"]],
        )

        rows = self._read_results(result_path)
        self.assertEqual(
            [row["scenario_id"] for row in rows],
            ["q-1", "q-1", "q-2", "q-2"],
        )
        self.assertEqual(
            [row["response"] for row in rows],
            ["Response 1", "Response 2", "Response 3", "Response 4"],
        )
        self.assertEqual([row["chatbot_id"] for row in rows], ["bot-a (fake)"] * 4)
        self.assertEqual(
            [row["input_prompt"] for row in rows],
            ["Prompt one", "Prompt one", "Prompt two", "Prompt two"],
        )
        self.assertEqual([row["error"] for row in rows], ["", "", "", ""])

    def test_includes_non_rag_results_after_rag_results_when_enabled(self) -> None:
        chatbot = FakeChatbot()
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]
        rag_prompts = ["RAG one", "RAG two"]
        non_rag_prompts = ["No RAG one", "No RAG two"]

        with (
            patch("utils.experiment.create_retriever", return_value=object()),
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch("utils.experiment.build_rag_prompts", return_value=rag_prompts),
            patch(
                "utils.experiment.build_non_rag_prompts",
                return_value=non_rag_prompts,
            ),
            patch("utils.experiment.create_chatbot", return_value=chatbot) as create,
        ):
            result_path = run_experiment(
                self._config(batch_size=2, include_non_rag=True),
                k=2,
            )

        create.assert_called_once()
        self.assertEqual(
            chatbot.batch_prompts,
            [
                ["RAG one", "RAG one"],
                ["RAG two", "RAG two"],
                ["No RAG one", "No RAG one"],
                ["No RAG two", "No RAG two"],
            ],
        )

        rows = self._read_results(result_path)
        self.assertEqual(
            [row["chatbot_id"] for row in rows],
            [
                "bot-a (fake)",
                "bot-a (fake)",
                "bot-a (fake)",
                "bot-a (fake)",
                "bot-a",
                "bot-a",
                "bot-a",
                "bot-a",
            ],
        )
        self.assertEqual(
            [row["scenario_id"] for row in rows],
            ["q-1", "q-1", "q-2", "q-2", "q-1", "q-1", "q-2", "q-2"],
        )
        self.assertEqual(
            [row["input_prompt"] for row in rows],
            [
                "RAG one",
                "RAG one",
                "RAG two",
                "RAG two",
                "No RAG one",
                "No RAG one",
                "No RAG two",
                "No RAG two",
            ],
        )

    def test_progress_total_doubles_when_non_rag_is_enabled(self) -> None:
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]

        with (
            patch("utils.experiment.create_retriever", return_value=object()),
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch("utils.experiment.build_rag_prompts", return_value=["RAG one", "RAG two"]),
            patch(
                "utils.experiment.build_non_rag_prompts",
                return_value=["No RAG one", "No RAG two"],
            ),
            patch("utils.experiment.create_chatbot", return_value=FakeChatbot()),
            patch("utils.experiment.tqdm", FakeTqdm),
        ):
            run_experiment(self._config(include_non_rag=True), k=3)

        self.assertEqual(FakeTqdm.instances[-1].total, 12)

    def test_initialization_failure_writes_error_rows_for_enabled_variants(self) -> None:
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]

        with (
            patch("utils.experiment.create_retriever", return_value=object()),
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch("utils.experiment.build_rag_prompts", return_value=["RAG one", "RAG two"]),
            patch(
                "utils.experiment.build_non_rag_prompts",
                return_value=["No RAG one", "No RAG two"],
            ),
            patch("utils.experiment.create_chatbot", side_effect=RuntimeError("boom")),
        ):
            result_path = run_experiment(self._config(include_non_rag=True))

        rows = self._read_results(result_path)
        self.assertEqual(
            [row["chatbot_id"] for row in rows],
            ["bot-a (fake)", "bot-a (fake)", "bot-a", "bot-a"],
        )
        self.assertEqual(
            [row["input_prompt"] for row in rows],
            ["Question one?", "Question two?", "Question one?", "Question two?"],
        )
        self.assertEqual(
            [row["error"] for row in rows],
            [
                "Failed to initialize chatbot: boom",
                "Failed to initialize chatbot: boom",
                "Failed to initialize chatbot: boom",
                "Failed to initialize chatbot: boom",
            ],
        )

    def test_failed_batch_falls_back_to_per_prompt_errors(self) -> None:
        chatbot = FakeChatbot(fail_batch=True, fail_prompt="Prompt two")
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]
        prompts = ["Prompt one", "Prompt two"]

        with (
            patch("utils.experiment.create_retriever", return_value=object()),
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch("utils.experiment.build_rag_prompts", return_value=prompts),
            patch("utils.experiment.create_chatbot", return_value=chatbot),
        ):
            result_path = run_experiment(self._config(batch_size=2))

        rows = self._read_results(result_path)
        self.assertEqual([row["scenario_id"] for row in rows], ["q-1", "q-2"])
        self.assertEqual([row["response"] for row in rows], ["Response 1", ""])
        self.assertEqual([row["chatbot_id"] for row in rows], ["bot-a (fake)", "bot-a (fake)"])
        self.assertEqual([row["input_prompt"] for row in rows], ["Prompt one", "Prompt two"])
        self.assertEqual([row["error"] for row in rows], ["", "single failure"])

    def test_resumes_partial_results_by_question_and_chatbot_id(self) -> None:
        chatbot = FakeChatbot()
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]
        self._write_results(
            [
                {
                    **eval_rows[0],
                    "chatbot_id": "bot-a (fake)",
                    "input_prompt": "Original prompt",
                    "response": "Original response",
                    "error": "",
                }
            ]
        )

        with (
            patch("utils.experiment.create_retriever", return_value=object()),
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch(
                "utils.experiment.build_rag_prompts",
                return_value=["Prompt one", "Prompt two"],
            ),
            patch("utils.experiment.create_chatbot", return_value=chatbot),
            patch("utils.experiment.tqdm", FakeTqdm),
        ):
            result_path = run_experiment(self._config(batch_size=3), k=2)

        rows = self._read_results(result_path)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["response"], "Original response")
        self.assertEqual(
            chatbot.batch_prompts,
            [["Prompt one", "Prompt two", "Prompt two"]],
        )
        self.assertEqual(FakeTqdm.instances[-1].total, 3)

    def test_retries_error_rows_without_duplicating_them(self) -> None:
        chatbot = FakeChatbot()
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]
        self._write_results(
            [
                {
                    **eval_rows[0],
                    "chatbot_id": "bot-a (fake)",
                    "input_prompt": "Original prompt",
                    "response": "Original response",
                    "error": "",
                },
                {
                    **eval_rows[1],
                    "chatbot_id": "bot-a (fake)",
                    "input_prompt": "Failed prompt",
                    "response": "",
                    "error": "Engine core initialization failed",
                },
            ]
        )

        with (
            patch("utils.experiment.create_retriever", return_value=object()),
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch(
                "utils.experiment.build_rag_prompts",
                return_value=["Retried prompt"],
            ) as build_prompts,
            patch("utils.experiment.create_chatbot", return_value=chatbot),
        ):
            result_path = run_experiment(self._config())

        rows = self._read_results(result_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["response"], "Original response")
        self.assertEqual(rows[1]["question"], "Question two?")
        self.assertEqual(rows[1]["input_prompt"], "Retried prompt")
        self.assertEqual(rows[1]["error"], "")
        build_prompts.assert_called_once_with(
            eval_rows=[eval_rows[1]],
            retriever=build_prompts.call_args.kwargs["retriever"],
            rag_config=self._config().rag_config,
            question_column="question",
        )

    def test_regenerates_stale_non_rag_prompt(self) -> None:
        chatbot = FakeChatbot()
        eval_row = {"scenario_id": "q-1", "question": "Question one?"}
        self._write_results(
            [
                {
                    **eval_row,
                    "chatbot_id": "bot-a (fake)",
                    "input_prompt": "RAG prompt",
                    "response": "RAG response",
                    "error": "",
                },
                {
                    **eval_row,
                    "chatbot_id": "bot-a",
                    "input_prompt": "Use the provided texts. Question: Question one?",
                    "response": "Stale response",
                    "error": "",
                },
            ]
        )

        with (
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=([eval_row], ["scenario_id", "question"]),
            ),
            patch("utils.experiment.create_chatbot", return_value=chatbot),
        ):
            result_path = run_experiment(self._config(include_non_rag=True))

        rows = self._read_results(result_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["response"], "RAG response")
        self.assertEqual(rows[1]["chatbot_id"], "bot-a")
        self.assertEqual(rows[1]["input_prompt"], "Question one?")
        self.assertEqual(chatbot.batch_prompts, [["Question one?"]])

    def test_complete_results_are_left_unchanged(self) -> None:
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]
        self._write_results(
            [
                {
                    **eval_row,
                    "chatbot_id": "bot-a (fake)",
                    "input_prompt": f"Prompt {index}",
                    "response": f"Response {index}",
                    "error": "",
                }
                for index, eval_row in enumerate(eval_rows, start=1)
            ]
        )
        result_path = self.out_dir / "results.csv"
        original_contents = result_path.read_bytes()

        with (
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch("utils.experiment.create_chatbot") as create_chatbot,
            patch("utils.experiment.create_retriever") as create_retriever,
            patch("utils.experiment.build_rag_prompts") as build_prompts,
        ):
            returned_path = run_experiment(self._config())

        self.assertEqual(returned_path, result_path)
        self.assertEqual(result_path.read_bytes(), original_contents)
        create_chatbot.assert_not_called()
        create_retriever.assert_not_called()
        build_prompts.assert_not_called()

    def test_rejects_invalid_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "`k` must be a positive integer"):
            run_experiment(self._config(), k=0)

    def test_runs_each_retriever_with_the_chatbot_it_serves(self) -> None:
        chatbot = FakeChatbot()
        retrievers = [
            RetrieverSpec("first", "FakeRetriever", {"mode": "first"}),
            RetrieverSpec("second", "FakeRetriever", {"mode": "second"}),
        ]
        eval_rows = [
            {"scenario_id": "q-1", "question": "Question one?"},
            {"scenario_id": "q-2", "question": "Question two?"},
        ]

        with (
            patch("utils.experiment.create_retriever", side_effect=[object(), object()]) as create,
            patch(
                "utils.experiment.load_eval_data_csv",
                return_value=(eval_rows, ["scenario_id", "question"]),
            ),
            patch(
                "utils.experiment.build_rag_prompts",
                side_effect=[["First one", "First two"], ["Second one", "Second two"]],
            ),
            patch("utils.experiment.create_chatbot", return_value=chatbot),
        ):
            result_path = run_experiment(self._config(retrievers=retrievers))

        create.assert_has_calls(
            [call(retrievers[0], chatbot), call(retrievers[1], chatbot)]
        )
        rows = self._read_results(result_path)
        self.assertEqual(
            [row["chatbot_id"] for row in rows],
            ["bot-a (first)", "bot-a (first)", "bot-a (second)", "bot-a (second)"],
        )
        self.assertEqual(
            [row["input_prompt"] for row in rows],
            ["First one", "First two", "Second one", "Second two"],
        )

    def _config(
        self,
        batch_size: int = 1,
        include_non_rag: bool = False,
        retrievers: list[RetrieverSpec] | None = None,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            out=str(self.out_dir),
            eval_data=EvalDataConfig(
                path="eval/scenarios.csv",
                question_column="question",
                ground_truth_columns=["answer"],
            ),
            rag_config=RagConfig(
                top_k=3,
                retrievers=retrievers
                or [RetrieverSpec("fake", "FakeRetriever", {})],
            ),
            chatbots=[
                ChatbotSpec(
                    id="bot-a",
                    backend="DummyChatbot",
                    config={},
                    batch_size=batch_size,
                )
            ],
            include_non_rag=include_non_rag,
        )

    def _write_results(self, rows: list[dict[str, str]]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
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
                    "chatbot_id",
                    "input_prompt",
                    "response",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _read_results(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as results_file:
            return list(csv.DictReader(results_file))


class FakeChatbot:
    def __init__(
        self,
        fail_batch: bool = False,
        fail_prompt: str | None = None,
    ) -> None:
        self.prompts: list[str] = []
        self.batch_prompts: list[list[str]] = []
        self.fail_batch = fail_batch
        self.fail_prompt = fail_prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None,
    ) -> tuple[str, list[str]]:
        if prompt == self.fail_prompt:
            raise RuntimeError("single failure")
        self.prompts.append(prompt)
        return f"Response {len(self.prompts)}", []

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int | None,
    ) -> list[tuple[str, list[str]]]:
        self.batch_prompts.append(prompts)
        if self.fail_batch:
            raise RuntimeError("batch failure")
        return [
            self.generate(prompt=prompt, max_new_tokens=max_new_tokens)
            for prompt in prompts
        ]


class FakeTqdm:
    instances: list["FakeTqdm"] = []

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str,
    ) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        FakeTqdm.instances.append(self)

    def __enter__(self) -> "FakeTqdm":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def update(self, n: int = 1) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
