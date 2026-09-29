import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from utils.config import ChatbotSpec, load_experiment_config
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
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Regenerate metrics and figures from the existing results CSV.",
    )
    return parser.parse_args()


def validate_hugging_face_access(chatbots: list[ChatbotSpec]) -> None:
    gated_chatbot_ids = [
        chatbot.id
        for chatbot in chatbots
        if chatbot.backend == "VLLMChatbot"
        and isinstance(chatbot.config.get("llm_kwargs"), dict)
        and chatbot.config["llm_kwargs"].get("hf_token") is True
    ]
    if not gated_chatbot_ids:
        return

    if _get_hugging_face_token():
        return

    chatbot_ids = ", ".join(gated_chatbot_ids)
    raise RuntimeError(
        "Hugging Face authentication is required for chatbot(s): "
        f"{chatbot_ids}. Set `HF_TOKEN` in `.env` and ensure the account has "
        "access to each gated model."
    )


def _get_hugging_face_token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

    try:
        # Load exp config
        experiment_config = load_experiment_config(args.experiment_json)

        if not args.skip_inference:
            validate_hugging_face_access(experiment_config.chatbots)
            result_path = run_experiment(experiment_config, k=args.k)
            tqdm.write(f"Saved experiment results: {result_path}")

        # Gaico eval
        gaico_paths = run_gaico(experiment_config)
        for gaico_path in gaico_paths:
            tqdm.write(f"Saved Gaico results: {gaico_path}")
        figure_paths = run_visualize(experiment_config)
        for figure_path in figure_paths:
            tqdm.write(f"Saved figure: {figure_path}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
