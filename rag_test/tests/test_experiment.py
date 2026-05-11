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
            ["scenario_id", "question", "chatbot_id", "response", "error"],
        )
        self.assertEqual(
            [row["scenario_id"] for row in rows],
            ["q-1", "q-1", "q-1", "q-2", "q-2", "q-2"],
        )

    def test_rejects_invalid_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "`k` must be a positive integer"):
            run_experiment(self._config(), k=0)

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

    def _read_results(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as results_file:
            return list(csv.DictReader(results_file))


class FakeChatbot:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None,
    ) -> tuple[str, list[str]]:
        self.prompts.append(prompt)
        return f"Response {len(self.prompts)}", []


if __name__ == "__main__":
    unittest.main()
