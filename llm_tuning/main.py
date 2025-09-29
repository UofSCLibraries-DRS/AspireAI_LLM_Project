import json
from dotenv import load_dotenv
import os
import argparse
from pathlib import Path

from utils.resources import ensure_pipeline
from utils.cache import TrainingCache, DummyCache


def main():
    # Parse args
    parser = argparse.ArgumentParser(description="My pipeline runner")
    parser.add_argument(
        "--pipeline-path",
        type=Path,
        required=True,
        help="Relative or absolute path to the pipeline file or directory",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable caching logic",
    )
    args = parser.parse_args()

    pipeline_path = args.pipeline_path.resolve()
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline path not found: {pipeline_path}")

    # Load .env
    load_dotenv()

    # Ensure everything is available
    ensure_pipeline(
        pipeline_path=pipeline_path,
        base_dir=os.getenv("BASE_DIR"),
    )

    # Create ordering of training steps
    # Wont be needed until later

    # Create cache object
    if args.cache:
        cache = TrainingCache("training_cache.json")
    else:
        cache = DummyCache()


if __name__ == "__main__":
    main()
