import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from utils.config import load_experiment_config
from utils.experiment import run_experiment
from utils.gaico import run_gaico
from utils.visualize import run_visualize


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None

    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a RAG experiment from an experiment config."
    )
    parser.add_argument(
        "experiment_json",
        help="Path to an experiment JSON file, e.g. configs/experiments/test.json",
    )
    parser.add_argument(
        "--k",
        type=positive_int,
        default=1,
        help="Number of times to prompt each model for each evaluation row.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    try:
        # Load exp config
        experiment_config = load_experiment_config(args.experiment_json)

        # Run exp
        # result_path = run_experiment(experiment_config, k=args.k)

        # Gaico eval
        gaico_paths = run_gaico(experiment_config)
        # print(f"Saved experiment results: {result_path}")
        for gaico_path in gaico_paths:
            print(f"Saved Gaico results: {gaico_path}")
        figure_paths = run_visualize(experiment_config)
        for figure_path in figure_paths:
            print(f"Saved figure: {figure_path}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
