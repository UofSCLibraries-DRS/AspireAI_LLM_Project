import json
import tempfile
import unittest
from pathlib import Path

from utils.config import load_experiment_config


class LoadExperimentConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_loads_new_eval_data_schema(self) -> None:
        config = load_experiment_config(self._write_config(_base_config()))

        self.assertEqual(config.out, "results/Exp1")
        self.assertEqual(config.data, "raw/combined.csv")
        self.assertEqual(config.eval_data.path, "eval/scenarios.csv")
        self.assertEqual(config.eval_data.question_column, "prompt")
        self.assertEqual(config.eval_data.ground_truth_columns, ["answer"])
        self.assertEqual(config.rag_config.embedding_model, "embedding-model")
        self.assertEqual(config.rag_config.top_k, 3)

    def test_eval_data_question_column_defaults_to_question(self) -> None:
        payload = _base_config()
        del payload["eval_data"]["question_column"]

        config = load_experiment_config(self._write_config(payload))

        self.assertEqual(config.eval_data.question_column, "question")

    def test_rejects_missing_eval_data(self) -> None:
        payload = _base_config()
        del payload["eval_data"]

        with self.assertRaisesRegex(ValueError, "Missing required object `eval_data`"):
            load_experiment_config(self._write_config(payload))

    def test_rejects_missing_eval_data_path(self) -> None:
        payload = _base_config()
        del payload["eval_data"]["path"]

        with self.assertRaisesRegex(ValueError, "Missing required string `path`"):
            load_experiment_config(self._write_config(payload))

    def test_rejects_invalid_ground_truth_columns(self) -> None:
        for invalid_value in ("answer", ["answer", ""], ["answer", 1], None):
            with self.subTest(invalid_value=invalid_value):
                payload = _base_config()
                payload["eval_data"]["ground_truth_columns"] = invalid_value

                with self.assertRaisesRegex(
                    ValueError,
                    "ground_truth_columns",
                ):
                    load_experiment_config(self._write_config(payload))

    def test_rejects_legacy_questions_without_eval_data(self) -> None:
        payload = _base_config()
        del payload["eval_data"]
        payload["questions"] = "eval/scenarios.csv"

        with self.assertRaisesRegex(ValueError, "Missing required object `eval_data`"):
            load_experiment_config(self._write_config(payload))

    def _write_config(self, payload: dict) -> Path:
        config_path = Path(self.temp_dir.name) / "experiment.json"
        with config_path.open("w", encoding="utf-8") as config_file:
            json.dump(payload, config_file)
        return config_path


def _base_config() -> dict:
    return {
        "out": "results/Exp1",
        "eval_data": {
            "path": "eval/scenarios.csv",
            "question_column": "prompt",
            "ground_truth_columns": ["answer"],
        },
        "data": "raw/combined.csv",
        "RAG_config": {
            "embedding_model": "embedding-model",
            "top_k": 3,
        },
        "chatbots": [
            {
                "id": "Dummy",
                "backend": "DummyChatbot",
                "config": {},
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
