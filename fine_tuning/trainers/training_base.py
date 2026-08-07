import os
import re
import shutil
import time
from datetime import timedelta
from pathlib import Path


CHECKPOINT_DIRECTORY = re.compile(r"^checkpoint-(\d+)$")


class AbstractTrainer:
    def __init__(
        self,
        start_model: str,
        output_dir: str,
        data: str,
        config: str,
        model_trace: list[str] | None = None,
        force_retrain: bool = False,
    ):
        self.start_model = start_model
        self.output_dir = output_dir
        self.data = data
        self.config = config
        self.model_trace = model_trace
        self.force_retrain = force_retrain

    @classmethod
    def subclass_by_name(cls, subclass_name, *args, **kwargs):
        for sc in cls.__subclasses__():
            if sc.__name__ == subclass_name:
                return sc(*args, **kwargs)
        raise ValueError(f"No such subclass of AbstractTrainer: {subclass_name}")

    def _train(self, resume_from_checkpoint: str | None = None):
        # To be overriden by base classes.
        # Does not handle trace logic
        pass

    def _checkpoint_directories(self) -> list[tuple[int, Path]]:
        """Return recognized Hugging Face checkpoints and their step numbers."""
        scratch_dir = Path(self.output_dir) / "scratch"
        if not scratch_dir.is_dir():
            return []

        checkpoints = []
        for path in scratch_dir.iterdir():
            match = CHECKPOINT_DIRECTORY.fullmatch(path.name)
            if match and path.is_dir():
                checkpoints.append((int(match.group(1)), path))
        return checkpoints

    def _prepare_checkpoint(self) -> str | None:
        """Select a resume checkpoint, or clear checkpoints for a forced run."""
        checkpoints = self._checkpoint_directories()

        if self.force_retrain:
            for _, checkpoint_path in checkpoints:
                if checkpoint_path.is_symlink():
                    checkpoint_path.unlink()
                else:
                    shutil.rmtree(checkpoint_path)
            if checkpoints:
                print(
                    f"Removed {len(checkpoints)} checkpoint(s) for forced "
                    "retraining."
                )
            print("Starting training from scratch.")
            return None

        if not checkpoints:
            print("No checkpoint found; starting training from scratch.")
            return None

        _, checkpoint_path = max(checkpoints, key=lambda item: item[0])
        checkpoint = str(checkpoint_path)
        print(f"Resuming training from checkpoint: {checkpoint}")
        return checkpoint

    def _model_exists(self):
        """Check if model exists by looking for HuggingFace files."""
        if not os.path.exists(self.output_dir):
            return False

        required_files = [
            "config.json",  # Model configuration
        ]

        # Check if config exists
        return any(
            os.path.exists(os.path.join(self.output_dir, f)) for f in required_files
        )

    def train(self):
        if self._model_exists() and not self.force_retrain:
            return
        os.makedirs(self.output_dir, exist_ok=True)

        resume_from_checkpoint = self._prepare_checkpoint()

        start = time.time()
        self._train(resume_from_checkpoint=resume_from_checkpoint)
        end = time.time()

        duration = timedelta(seconds=end - start)

        end_message = (
            f"Training finished at: {time.ctime(end)}\n"
            f"Total training duration: {duration}"
        )
        print(end_message)

        if not self.model_trace:
            return
        # Write model trace to a file in the output_dir
        trace_path = os.path.join(self.output_dir, "model_trace.txt")
        with open(trace_path, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in self.model_trace)
