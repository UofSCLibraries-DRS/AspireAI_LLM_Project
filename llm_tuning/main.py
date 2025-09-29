from dotenv import load_dotenv
import os
import argparse
from pathlib import Path

from utils.resources import ensure_pipeline


def main():
    # Parse args
    parser = argparse.ArgumentParser(description="My pipeline runner")
    parser.add_argument(
        "--pipeline-path",
        type=Path,
        required=True,
        help="Relative or absolute path to the pipeline file or directory",
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

    # Create cache file

    return None


if __name__ == "__main__":
    main()
