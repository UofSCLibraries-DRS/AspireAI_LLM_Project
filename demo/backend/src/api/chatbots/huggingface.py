from typing import Optional
from dataclasses import dataclass
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .base import Chatbot


@dataclass
class HuggingFaceChatbotConfig:
    model_path: str
    model_temperature: float
    prompt_template_path: str

    @classmethod
    def from_yaml(cls, path: str) -> "HuggingFaceChatbotConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML file must contain a mapping at the top level")

        required_fields = {f.name for f in cls.__dataclass_fields__.values()}
        missing = required_fields - data.keys()

        if missing:
            raise ValueError(f"Missing required config fields in `{path}`: {missing}")

        return cls(**data)


class HuggingFaceChatbot(Chatbot):
    """
    SafeChat wrapper.
    """

    def __init__(
        self,
        id: str,
        config_path: str,
    ):
        super().__init__(id=id)

        # Read config
        self.config = HuggingFaceChatbotConfig.from_yaml(config_path)

        # Get stop sequences and template from template config
        with open(self.config.prompt_template_path, "r") as f:
            prompt_template_cfg = yaml.safe_load(f)

        self.prompt_template = prompt_template_cfg.get(
            "template", "{user_prompt}"
        )  # Default to identity function
        self.stop_sequences = prompt_template_cfg.get("stop_sequences", [])

        # Initialize tokenizer and model
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
        text = text[:min_idx].strip()
        return text

    def generate(
        self, prompt: str, max_new_tokens: Optional[int] = 128
    ) -> (str, list[str]):
        templated_prompt = self.prompt_template.format(user_prompt=prompt)

        # Tokenize and generate
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

        # Decode output
        full_text = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Post-process (trim at stop sequences)
        return self._post_process(full_text, templated_prompt), []
