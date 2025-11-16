from typing import List
import os
import torch
import gc
from accelerate import Accelerator
import time
from datetime import timedelta


class AbstractTrainer:
    def __init__(
        self,
        start_model: str,
        output_dir: str,
        data: str,
        config: str,
        model_trace: List[str] | None = None,
    ):
        self.start_model = start_model
        self.output_dir = output_dir
        self.data = data
        self.config = config
        self.model_trace = model_trace

    @classmethod
    def subclass_by_name(cls, subclass_name, *args, **kwargs):
        for sc in cls.__subclasses__():
            if sc.__name__ == subclass_name:
                return sc(*args, **kwargs)
        raise ValueError(f"No such subclass of AbstractTrainer: {subclass_name}")

    def _train(self):
        # To be overriden by base classes.
        # Does not handle trace logic
        pass

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)
        start = time.time()
        self._train()
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
            for line in self.model_trace:
                f.write(line + "\n")
