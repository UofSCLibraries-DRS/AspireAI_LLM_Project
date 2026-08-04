import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jsonschema import ValidationError

from fine_tuning.utils.cache.hash import list_hash
from fine_tuning.utils.validation import (
    validate_pipeline_json,
    validate_training_step_json,
)


class PipelineHashTest(unittest.TestCase):
    def test_hashes_inline_steps_stably_across_key_order(self):
        first = [{"trainer": "Trainer", "data": "data.csv", "config": "config.json"}]
        second = [{"config": "config.json", "trainer": "Trainer", "data": "data.csv"}]

        self.assertEqual(list_hash(first), list_hash(second))

    def test_hash_distinguishes_sequence_boundaries(self):
        self.assertNotEqual(list_hash(["ab", "c"]), list_hash(["a", "bc"]))

    def test_pipeline_schema_accepts_inline_steps(self):
        step = {
            "trainer": "Trainer",
            "data": "data.csv",
            "config": "config.json",
        }
        pipeline = [
            {
                "model": {
                    "start": "base",
                    "train_steps": [step],
                    "output": "trained",
                }
            }
        ]

        validate_pipeline_json(pipeline)
        validate_training_step_json(step)

    def test_pipeline_schema_rejects_legacy_string_steps(self):
        pipeline = [
            {
                "model": {
                    "start": "base",
                    "train_steps": ["legacy-step.json"],
                    "output": "trained",
                }
            }
        ]

        with self.assertRaises(ValidationError):
            validate_pipeline_json(pipeline)

    def test_training_step_schema_requires_all_fields(self):
        incomplete_step = {
            "trainer": "Trainer",
            "data": "data.csv",
        }

        with self.assertRaises(ValidationError):
            validate_training_step_json(incomplete_step)

    def test_reference_pipeline_files_pass_validation(self):
        pipeline_dir = Path(__file__).parents[1] / "config" / "pipelines" / "llama"

        for filename in ("M13.json", "M12_full.json"):
            with self.subTest(filename=filename):
                pipeline = json.loads(
                    (pipeline_dir / filename).read_text(encoding="utf-8")
                )
                validate_pipeline_json(pipeline)

    def test_build_pipeline_reuses_a_shared_inline_training_prefix(self):
        step_a = {
            "trainer": "LoRASFTTrainer",
            "data": "a.csv",
            "config": "a.json",
        }
        step_b = {
            "trainer": "LoRASFTTrainer",
            "data": "b.csv",
            "config": "b.json",
        }
        pipelines = [
            {
                "model": {
                    "start": "base",
                    "train_steps": [step_a],
                    "output": "model-a",
                },
                "inference": [],
            },
            {
                "model": {
                    "start": "base",
                    "train_steps": [step_a, step_b],
                    "output": "model-ab",
                },
                "inference": [],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_folder = root / "models"
            model_dump = root / "dump"
            data_folder = root / "data"
            config_folder = root / "config"
            prompt_folder = root / "prompts"
            for directory in (
                model_folder / "base",
                model_dump,
                data_folder,
                config_folder,
                prompt_folder,
            ):
                directory.mkdir(parents=True, exist_ok=True)
            pipeline_path = root / "pipeline.json"
            pipeline_path.write_text(json.dumps(pipelines), encoding="utf-8")

            environment = {
                "MODEL_FOLDER": str(model_folder),
                "MODEL_DUMP": str(model_dump),
                "DATA_FOLDER": str(data_folder),
                "CONFIG_FOLDER": str(config_folder),
                "PROMPT_FOLDER": str(prompt_folder),
            }
            trainer_module = types.ModuleType("fine_tuning.trainers")
            inference_module = types.ModuleType("fine_tuning.inference")

            class StubTrainer:
                @classmethod
                def subclass_by_name(cls, subclass_name, **kwargs):
                    return SimpleNamespace(trainer=subclass_name, **kwargs)

            trainer_module.AbstractTrainer = StubTrainer
            inference_module.InferenceJob = SimpleNamespace

            sys.modules.pop("fine_tuning.utils.parse_pipeline", None)
            try:
                with (
                    patch.dict(os.environ, environment),
                    patch.dict(
                        sys.modules,
                        {
                            "fine_tuning.trainers": trainer_module,
                            "fine_tuning.inference": inference_module,
                        },
                    ),
                ):
                    parse_pipeline = importlib.import_module(
                        "fine_tuning.utils.parse_pipeline"
                    )
                    result = parse_pipeline.build_pipeline(str(pipeline_path))
            finally:
                sys.modules.pop("fine_tuning.utils.parse_pipeline", None)

        self.assertEqual(len(result.train_steps), 2)
        first, second = result.train_steps
        self.assertEqual(first.start_model, str(model_folder / "base"))
        self.assertEqual(first.output_dir, str(model_folder / "model-a"))
        self.assertEqual(second.start_model, str(model_folder / "model-a"))
        self.assertEqual(second.output_dir, str(model_folder / "model-ab"))


if __name__ == "__main__":
    unittest.main()
