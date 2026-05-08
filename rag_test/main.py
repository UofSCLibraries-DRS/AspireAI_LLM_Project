import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from utils.config import load_experiment_config
from utils.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a RAG experiment from an experiment config."
    )
    parser.add_argument(
        "experiment_json",
        help="Path to an experiment JSON file, e.g. configs/experiments/test.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    try:
        # Load exp config
        experiment_config = load_experiment_config(args.experiment_json)

        # Run exp
        result_path = run_experiment(experiment_config)

        # Gaico eval
        print(f"Saved experiment results: {result_path}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
