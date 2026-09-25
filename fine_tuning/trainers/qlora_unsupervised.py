"""QLoRA training for pre-rendered causal-language-model text datasets.

QLoRA keeps the base model in 4-bit NF4 format and trains only LoRA adapter
weights. Consequently the output directory is an adapter checkpoint, not a
merged full-precision model.
"""

import gc
import json
import os

import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from .lora_unsupervised import causal_lm_collator
from .training_base import AbstractTrainer
from fine_tuning.utils.causal_lm import (
    configure_padding_token,
    tokenize_causal_lm_batch,
)
from fine_tuning.utils.training_data import MAX_SEQUENCE_LENGTH, load_text_column


def _compute_dtype(name: str | None) -> torch.dtype:
    """Resolve the optional JSON dtype name to a torch dtype."""
    if name is None:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    try:
        return dtypes[name.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            "quantization_config.compute_dtype must be one of "
            "'bfloat16', 'bf16', 'float16', or 'fp16'."
        ) from error


def qlora_quantization_config(config: dict) -> BitsAndBytesConfig:
    """Build the fixed 4-bit NF4 quantization configuration used by QLoRA."""
    quantization = config.get("quantization_config", {})
    if quantization.get("load_in_4bit", True) is not True:
        raise ValueError("QLoRA requires quantization_config.load_in_4bit to be true.")

    quantization_type = quantization.get("bnb_4bit_quant_type", "nf4")
    if quantization_type.lower() != "nf4":
        raise ValueError("QLoRA requires quantization_config.bnb_4bit_quant_type 'nf4'.")

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=quantization.get(
            "bnb_4bit_use_double_quant", True
        ),
        bnb_4bit_compute_dtype=_compute_dtype(
            quantization.get(
                "compute_dtype", quantization.get("bnb_4bit_compute_dtype")
            )
        ),
    )


class QLoRAUnsupervisedTrainer(AbstractTrainer):
    """Fine-tune a 4-bit causal LM with LoRA adapters from a CSV ``text`` column.

    The config has the same ``lora_config`` and ``training_args`` sections as
    ``LoRAUnsupervisedTrainer``. Its optional ``quantization_config`` section
    accepts ``compute_dtype`` and ``bnb_4bit_use_double_quant``; QLoRA always
    uses 4-bit NF4 quantization.
    """

    def _train(self, resume_from_checkpoint: str | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA training requires a CUDA-capable GPU.")

        log_dir = os.path.join(self.output_dir, "logs")
        scratch_dir = os.path.join(self.output_dir, "scratch")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(scratch_dir, exist_ok=True)

        with open(self.config, encoding="utf-8") as handle:
            config = json.load(handle)

        text_values = load_text_column(self.data)
        dataset = Dataset.from_dict({"text": text_values})

        tokenizer = AutoTokenizer.from_pretrained(self.start_model)
        configure_padding_token(tokenizer)

        def tokenize(batch):
            return tokenize_causal_lm_batch(
                tokenizer,
                batch["text"],
                max_length=MAX_SEQUENCE_LENGTH,
            )

        tokenized_dataset = dataset.map(
            tokenize, batched=True, remove_columns=["text"]
        )

        quantization_config = qlora_quantization_config(config)
        model = AutoModelForCausalLM.from_pretrained(
            self.start_model,
            quantization_config=quantization_config,
            torch_dtype=quantization_config.bnb_4bit_compute_dtype,
            device_map={"": Accelerator().local_process_index},
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
        model = get_peft_model(model, LoraConfig(**config["lora_config"]))

        training_config = dict(config["training_args"])
        training_config.setdefault("optim", "paged_adamw_8bit")
        training_config.setdefault("gradient_checkpointing", True)
        training_config.setdefault(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        )
        training_args = TrainingArguments(
            **training_config,
            output_dir=scratch_dir,
            logging_dir=log_dir,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=tokenizer,
            data_collator=causal_lm_collator,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Saving at output_dir makes this adapter a usable pipeline artifact.
        # Do not merge: a 4-bit merge needs a higher-precision base reload.
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        with open(
            os.path.join(log_dir, "log_history.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(trainer.state.log_history, handle, indent=2)

        losses = [
            entry["loss"] for entry in trainer.state.log_history if "loss" in entry
        ]
        steps = [
            entry["step"] for entry in trainer.state.log_history if "loss" in entry
        ]
        if len(losses) > 1:
            plt.figure(figsize=(8, 5))
            plt.plot(steps, losses, label="Training Loss", linewidth=2)
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("QLoRA Training Loss Curve")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, "loss_plot.png"))
            plt.close()
        else:
            print("No loss data found in trainer.state.log_history — skipping plot.")

        del model, trainer, tokenizer, dataset, tokenized_dataset
        gc.collect()
        torch.cuda.empty_cache()
        Accelerator().free_memory()
