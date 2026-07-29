"""Inline training-step JSON files in a pipeline configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"


def flatten_training_config(pipeline: Any, config_root: Path) -> Any:
    """Replace string training steps with the JSON documents they reference."""
    if not isinstance(pipeline, list):
        raise ValueError("pipeline JSON must contain a top-level array")

    for row in pipeline:
        if not isinstance(row, dict):
            continue

        model = row.get("model")
        if not isinstance(model, dict):
            continue

        train_steps = model.get("train_steps")
        if not isinstance(train_steps, list):
            continue

        flattened_steps = []
        for train_step in train_steps:
            if not isinstance(train_step, str):
                flattened_steps.append(train_step)
                continue

            train_step_path = config_root / train_step
            try:
                with train_step_path.open(encoding="utf-8") as step_file:
                    flattened_steps.append(json.load(step_file))
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"training-step config not found: {train_step_path}"
                ) from error
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in training-step config {train_step_path}: {error}"
                ) from error

        model["train_steps"] = flattened_steps

    return pipeline


def convert_file(input_path: Path, output_path: Path, config_root: Path) -> None:
    """Load, flatten, and write a pipeline configuration."""
    try:
        with input_path.open(encoding="utf-8") as pipeline_file:
            pipeline = json.load(pipeline_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid pipeline JSON in {input_path}: {error}") from error

    flattened_pipeline = flatten_training_config(pipeline, config_root)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(flattened_pipeline, output_file, indent=4)
        output_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inline JSON files referenced by pipeline train_steps."
    )
    parser.add_argument("input_path", type=Path, help="Pipeline JSON to convert")
    parser.add_argument("output_path", type=Path, help="Converted pipeline JSON")
    parser.add_argument(
        "--config-root",
        type=Path,
        default=DEFAULT_CONFIG_ROOT,
        help=f"Root for training-step paths (default: {DEFAULT_CONFIG_ROOT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        convert_file(args.input_path, args.output_path, args.config_root)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
