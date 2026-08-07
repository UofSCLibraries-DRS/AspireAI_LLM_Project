import ast
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


FINE_TUNING_DIR = Path(__file__).parents[1]
TRAINING_BASE_PATH = FINE_TUNING_DIR / "trainers" / "training_base.py"

spec = importlib.util.spec_from_file_location(
    "training_base_under_test", TRAINING_BASE_PATH
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {TRAINING_BASE_PATH}")
training_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_base)
AbstractTrainer = training_base.AbstractTrainer


class FakeHuggingFaceTrainer:
    def __init__(self):
        self.resume_from_checkpoint = "not-called"

    def train(self, resume_from_checkpoint: str | None = None):
        self.resume_from_checkpoint = resume_from_checkpoint


class RecordingTrainer(AbstractTrainer):
    def __init__(self, output_dir: str, force_retrain: bool = False):
        super().__init__(
            start_model="start-model",
            output_dir=output_dir,
            data="data.csv",
            config="config.json",
            force_retrain=force_retrain,
        )
        self.huggingface_trainer = FakeHuggingFaceTrainer()

    @property
    def resume_from_checkpoint(self):
        return self.huggingface_trainer.resume_from_checkpoint

    def _train(self, resume_from_checkpoint: str | None = None):
        self.huggingface_trainer.train(resume_from_checkpoint=resume_from_checkpoint)


class FailingRecordingTrainer(RecordingTrainer):
    def _train(self, resume_from_checkpoint: str | None = None):
        super()._train(resume_from_checkpoint=resume_from_checkpoint)
        raise RuntimeError("invalid checkpoint")


class CheckpointResumeTest(unittest.TestCase):
    def run_trainer(self, trainer: RecordingTrainer) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            trainer.train()
        return output.getvalue()

    def test_missing_or_empty_scratch_directory_starts_fresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for create_scratch in (False, True):
                with self.subTest(create_scratch=create_scratch):
                    output_dir = Path(temp_dir) / str(create_scratch)
                    if create_scratch:
                        (output_dir / "scratch").mkdir(parents=True)

                    trainer = RecordingTrainer(str(output_dir))
                    output = self.run_trainer(trainer)

                    self.assertIsNone(trainer.resume_from_checkpoint)
                    self.assertIn(
                        "No checkpoint found; starting training from scratch.", output
                    )

    def test_latest_checkpoint_is_selected_by_numeric_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scratch_dir = Path(temp_dir) / "scratch"
            for checkpoint in ("checkpoint-20", "checkpoint-3", "checkpoint-100"):
                (scratch_dir / checkpoint).mkdir(parents=True)

            trainer = RecordingTrainer(temp_dir)
            output = self.run_trainer(trainer)
            expected = str(scratch_dir / "checkpoint-100")

            self.assertEqual(trainer.resume_from_checkpoint, expected)
            self.assertIn(f"Resuming training from checkpoint: {expected}", output)

    def test_unrecognized_entries_are_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scratch_dir = Path(temp_dir) / "scratch"
            (scratch_dir / "checkpoint-4").mkdir(parents=True)
            (scratch_dir / "checkpoint-final").mkdir()
            (scratch_dir / "checkpoint-12-extra").mkdir()
            (scratch_dir / "unrelated").mkdir()
            (scratch_dir / "checkpoint-999").write_text("not a directory")

            trainer = RecordingTrainer(temp_dir)
            self.run_trainer(trainer)

            self.assertEqual(
                trainer.resume_from_checkpoint, str(scratch_dir / "checkpoint-4")
            )

    def test_invalid_latest_checkpoint_fails_without_falling_back(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scratch_dir = Path(temp_dir) / "scratch"
            (scratch_dir / "checkpoint-5").mkdir(parents=True)
            (scratch_dir / "checkpoint-10").mkdir()

            trainer = FailingRecordingTrainer(temp_dir)
            with self.assertRaisesRegex(RuntimeError, "invalid checkpoint"):
                self.run_trainer(trainer)

            self.assertEqual(
                trainer.resume_from_checkpoint, str(scratch_dir / "checkpoint-10")
            )

    def test_force_retrain_removes_only_recognized_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scratch_dir = Path(temp_dir) / "scratch"
            checkpoint_paths = [
                scratch_dir / "checkpoint-2",
                scratch_dir / "checkpoint-15",
            ]
            for path in checkpoint_paths:
                path.mkdir(parents=True)
                (path / "trainer_state.json").write_text("{}")

            malformed_dir = scratch_dir / "checkpoint-latest"
            malformed_dir.mkdir()
            unrelated_file = scratch_dir / "notes.txt"
            unrelated_file.write_text("keep me")

            trainer = RecordingTrainer(temp_dir, force_retrain=True)
            output = self.run_trainer(trainer)

            self.assertIsNone(trainer.resume_from_checkpoint)
            self.assertTrue(all(not path.exists() for path in checkpoint_paths))
            self.assertTrue(malformed_dir.is_dir())
            self.assertEqual(unrelated_file.read_text(), "keep me")
            self.assertIn("Removed 2 checkpoint(s) for forced retraining.", output)
            self.assertIn("Starting training from scratch.", output)

    def test_completed_model_skips_checkpoint_handling_unless_forced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "config.json").write_text("{}")
            checkpoint_dir = output_dir / "scratch" / "checkpoint-8"
            checkpoint_dir.mkdir(parents=True)

            trainer = RecordingTrainer(temp_dir)
            output = self.run_trainer(trainer)

            self.assertEqual(trainer.resume_from_checkpoint, "not-called")
            self.assertTrue(checkpoint_dir.is_dir())
            self.assertEqual(output, "")

            forced_trainer = RecordingTrainer(temp_dir, force_retrain=True)
            self.run_trainer(forced_trainer)

            self.assertIsNone(forced_trainer.resume_from_checkpoint)
            self.assertFalse(checkpoint_dir.exists())


class TrainerCheckpointIntegrationTest(unittest.TestCase):
    TRAINERS = {
        "full_sft.py": "FullSFTTrainer",
        "lora_sft.py": "LoRASFTTrainer",
        "full_unsupervised.py": "FullUnsupervisedTrainer",
        "lora_unsupervised.py": "LoRAUnsupervisedTrainer",
    }

    def test_all_trainers_forward_the_resume_checkpoint(self):
        trainers_dir = FINE_TUNING_DIR / "trainers"

        for filename, class_name in self.TRAINERS.items():
            with self.subTest(trainer=class_name):
                tree = ast.parse((trainers_dir / filename).read_text(encoding="utf-8"))
                trainer_class = next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.ClassDef) and node.name == class_name
                )
                train_method = next(
                    node
                    for node in trainer_class.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_train"
                )

                self.assertIn(
                    "resume_from_checkpoint",
                    [argument.arg for argument in train_method.args.args],
                )

                trainer_train_calls = [
                    node
                    for node in ast.walk(train_method)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "trainer"
                    and node.func.attr == "train"
                ]
                self.assertEqual(len(trainer_train_calls), 1)

                resume_keywords = [
                    keyword
                    for keyword in trainer_train_calls[0].keywords
                    if keyword.arg == "resume_from_checkpoint"
                ]
                self.assertEqual(len(resume_keywords), 1)
                self.assertIsInstance(resume_keywords[0].value, ast.Name)
                self.assertEqual(resume_keywords[0].value.id, "resume_from_checkpoint")


if __name__ == "__main__":
    unittest.main()
