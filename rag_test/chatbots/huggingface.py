from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Chatbot
from .config import config_from_mapping_or_yaml


@dataclass
class HuggingFaceChatbotConfig:
    model_path: str
    prompt_template_path: str
    model_temperature: float = 0.5

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str | Path,
    ) -> "HuggingFaceChatbotConfig":
        return config_from_mapping_or_yaml(
            config_cls=cls,
            config=config,
            aliases={
                "temperature": "model_temperature",
                "chat_template_path": "prompt_template_path",
            },
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HuggingFaceChatbotConfig":
        return cls.from_config(path)


class HuggingFaceChatbot(Chatbot):
    """
    Hugging Face causal language model chatbot wrapper.
    """

    def __init__(
        self,
        id: str,
        config: Mapping[str, Any] | str | Path,
    ):
        super().__init__(id=id)
        self.config = HuggingFaceChatbotConfig.from_config(config)

        with open(self.config.prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template_cfg = yaml.safe_load(f)

        self.prompt_template = prompt_template_cfg.get("template", "{user_prompt}")
        self.stop_sequences = prompt_template_cfg.get("stop_sequences", [])

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype="auto",
            device_map="auto",
        )

    def _post_process(self, text: str, templated_prompt: str) -> str:
        text = text.removeprefix(templated_prompt)

        min_idx = len(text)
        for stop in self.stop_sequences:
            idx = text.find(stop)
            if idx != -1:
                min_idx = min(idx, min_idx)
        return text[:min_idx].strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = 128,
    ) -> tuple[str, list[str]]:
        templated_prompt = self.prompt_template.format(user_prompt=prompt)

        inputs = self.tokenizer(templated_prompt, return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.config.model_temperature,
            )

        full_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return self._post_process(full_text, templated_prompt), []
