import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.config import ChatbotSpec, EvalDataConfig, ExperimentConfig, RagConfig
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
            patch(
                "utils.experiment._ensure_embedding_cache",
                return_value=Path("cache"),
            ),
            patch("utils.experiment.ensure_faiss_index", return_value=object()),
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
        self.assertEqual({row["chatbot_id"] for row in rows}, {"bot-a (RAG)"})
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
            patch(
                "utils.experiment._ensure_embedding_cache",
                return_value=Path("cache"),
            ),
            patch("utils.experiment.ensure_faiss_index", return_value=object()),
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
        self.assertEqual([row["chatbot_id"] for row in rows], ["bot-a (RAG)"] * 4)
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
            patch(
                "utils.experiment._ensure_embedding_cache",
                return_value=Path("cache"),
            ),
            patch("utils.experiment.ensure_faiss_index", return_value=object()),
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
                "bot-a (RAG)",
                "bot-a (RAG)",
                "bot-a (RAG)",
                "bot-a (RAG)",
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
            patch(
                "utils.experiment._ensure_embedding_cache",
                return_value=Path("cache"),
            ),
            patch("utils.experiment.ensure_faiss_index", return_value=object()),
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
            patch(
                "utils.experiment._ensure_embedding_cache",
                return_value=Path("cache"),
            ),
            patch("utils.experiment.ensure_faiss_index", return_value=object()),
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
            ["bot-a (RAG)", "bot-a (RAG)", "bot-a", "bot-a"],
        )
        self.assertEqual(
            [row["input_prompt"] for row in rows],
            ["RAG one", "RAG two", "No RAG one", "No RAG two"],
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
            patch(
                "utils.experiment._ensure_embedding_cache",
                return_value=Path("cache"),
            ),
            patch("utils.experiment.ensure_faiss_index", return_value=object()),
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
        self.assertEqual([row["chatbot_id"] for row in rows], ["bot-a (RAG)", "bot-a (RAG)"])
        self.assertEqual([row["input_prompt"] for row in rows], ["Prompt one", "Prompt two"])
        self.assertEqual([row["error"] for row in rows], ["", "single failure"])

    def test_rejects_invalid_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "`k` must be a positive integer"):
            run_experiment(self._config(), k=0)

    def _config(
        self,
        batch_size: int = 1,
        include_non_rag: bool = False,
    ) -> ExperimentConfig:
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
                    batch_size=batch_size,
                )
            ],
            include_non_rag=include_non_rag,
        )

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
