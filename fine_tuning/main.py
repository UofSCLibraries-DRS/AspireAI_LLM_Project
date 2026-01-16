from fine_tuning.evaluation import gaico_accuracy
from dotenv import load_dotenv, find_dotenv
import os
import argparse
from pathlib import Path

from fine_tuning.utils.parse_pipeline import build_pipeline
from fine_tuning.inference import batched_inference


def main():
    print("here")
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline-path",
        type=Path,
        required=True,
        help="Absolute path to the pipeline file or directory",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help=".env file to use",
    )
    args = parser.parse_args()

    print("Testing std out")

    # Ensure pipeline config path exists
    pipeline_path = args.pipeline_path.resolve()
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline path not found: {pipeline_path}")

    # Load env file
    if os.path.isabs(args.env):
        load_dotenv(args.env)
    else:
        load_dotenv(find_dotenv(args.env))

    # Build the pipeline
    pipeline = build_pipeline(pipeline_path)

    print("Finished building pipeline")

    # Run all training steps
    for train_step in pipeline.train_steps:
        train_step.train()

    inf_results = batched_inference(pipeline.inference_jobs)
    gaico_accuracy(inf_results)


if __name__ == "__main__":
    main()
