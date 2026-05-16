import argparse
import unittest
from pathlib import Path
from unittest.mock import call, patch

from main import main, positive_int
from utils.config import ExperimentConfig


class MainCliTests(unittest.TestCase):
    def test_positive_int_accepts_positive_values(self) -> None:
        self.assertEqual(positive_int("3"), 3)

    def test_positive_int_rejects_non_positive_values(self) -> None:
        for value in ("0", "-1", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_int(value)

    def test_main_runs_full_pipeline_in_order(self) -> None:
        experiment_config = object()
        calls = []

        def fake_load_experiment_config(path: str) -> object:
            calls.append(("load", path))
            return experiment_config

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
            )),
            patch("main.load_dotenv"),
            patch("main.load_experiment_config", side_effect=fake_load_experiment_config),
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


if __name__ == "__main__":
    unittest.main()
