import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from main import main, positive_int, validate_hugging_face_access
from utils.config import ChatbotSpec, ExperimentConfig


class MainCliTests(unittest.TestCase):
    def test_positive_int_accepts_positive_values(self) -> None:
        self.assertEqual(positive_int("3"), 3)

    def test_positive_int_rejects_non_positive_values(self) -> None:
        for value in ("0", "-1", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_int(value)

    def test_main_runs_full_pipeline_in_order(self) -> None:
        experiment_config = SimpleNamespace(chatbots=[])
        calls = []

        def fake_load_experiment_config(path: str) -> object:
            calls.append(("load", path))
            return experiment_config

        def fake_validate_hugging_face_access(
            chatbots: list[ChatbotSpec],
        ) -> None:
            calls.append(("validate", chatbots))

        def fake_run_experiment(config: ExperimentConfig, k: int) -> Path:
            calls.append(("experiment", config, k))
            return Path("results/Exp1/results.csv")

        def fake_run_gaico(config: ExperimentConfig) -> list[Path]:
            calls.append(("gaico", config))
            return [Path("results/Exp1/gaico/answer.csv")]

        def fake_run_visualize(config: ExperimentConfig) -> list[Path]:
            calls.append(("visualize", config))
            return [Path("results/Exp1/figures/answer_radar.png")]

        with (
            patch("main.parse_args", return_value=argparse.Namespace(
                experiment_json="configs/experiments/test.json",
                k=3,
                skip_inference=False,
            )),
            patch("main.load_dotenv"),
            patch("main.load_experiment_config", side_effect=fake_load_experiment_config),
            patch(
                "main.validate_hugging_face_access",
                side_effect=fake_validate_hugging_face_access,
            ),
            patch("main.run_experiment", side_effect=fake_run_experiment),
            patch("main.run_gaico", side_effect=fake_run_gaico),
            patch("main.run_visualize", side_effect=fake_run_visualize),
            patch("main.tqdm.write") as write,
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            calls,
            [
                ("load", "configs/experiments/test.json"),
                ("validate", []),
                ("experiment", experiment_config, 3),
                ("gaico", experiment_config),
                ("visualize", experiment_config),
            ],
        )
        write.assert_has_calls(
            [
                call("Saved experiment results: results/Exp1/results.csv"),
                call("Saved Gaico results: results/Exp1/gaico/answer.csv"),
                call("Saved figure: results/Exp1/figures/answer_radar.png"),
            ]
        )

    def test_skip_inference_runs_metrics_and_figures_only(self) -> None:
        experiment_config = SimpleNamespace(chatbots=[])

        with (
            patch(
                "main.parse_args",
                return_value=argparse.Namespace(
                    experiment_json="configs/experiments/test.json",
                    k=1,
                    skip_inference=True,
                ),
            ),
            patch("main.load_dotenv"),
            patch(
                "main.load_experiment_config",
                return_value=experiment_config,
            ),
            patch("main.validate_hugging_face_access") as validate,
            patch("main.run_experiment") as run_inference,
            patch("main.run_gaico", return_value=[]) as run_metrics,
            patch("main.run_visualize", return_value=[]) as run_figures,
        ):
            exit_code = main()

        self.assertEqual(exit_code, 0)
        validate.assert_not_called()
        run_inference.assert_not_called()
        run_metrics.assert_called_once_with(experiment_config)
        run_figures.assert_called_once_with(experiment_config)

    def test_hugging_face_preflight_rejects_missing_token(self) -> None:
        chatbot = ChatbotSpec(
            id="meta",
            backend="VLLMChatbot",
            config={"llm_kwargs": {"hf_token": True}},
        )

        with patch("main._get_hugging_face_token", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                validate_hugging_face_access([chatbot])

    def test_hugging_face_preflight_accepts_available_token(self) -> None:
        chatbot = ChatbotSpec(
            id="meta",
            backend="VLLMChatbot",
            config={"llm_kwargs": {"hf_token": True}},
        )

        with patch("main._get_hugging_face_token", return_value="token"):
            validate_hugging_face_access([chatbot])

    def test_hugging_face_preflight_ignores_models_without_hf_token(self) -> None:
        chatbot = ChatbotSpec(
            id="local",
            backend="VLLMChatbot",
            config={"llm_kwargs": {}},
        )

        with patch("main._get_hugging_face_token") as get_token:
            validate_hugging_face_access([chatbot])

        get_token.assert_not_called()


if __name__ == "__main__":
    unittest.main()
