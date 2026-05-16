from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .base import Chatbot
from .config import config_from_mapping_or_yaml


@dataclass
class VLLMChatbotConfig:
    model_path: str
    prompt_template_path: str | None = None
    model_temperature: float = 0.5
    max_tokens: int = 128
    top_p: float = 1.0
    llm_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.llm_kwargs, dict):
            raise ValueError("`llm_kwargs` must be an object")
        if "model" in self.llm_kwargs:
            raise ValueError("`llm_kwargs` must not contain `model`; use `model_path`")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str | Path,
    ) -> "VLLMChatbotConfig":
        return config_from_mapping_or_yaml(
            config_cls=cls,
            config=config,
            aliases={
                "temperature": "model_temperature",
                "chat_template_path": "prompt_template_path",
                "model_id": "model_path",
                "model": "model_path",
            },
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VLLMChatbotConfig":
        return cls.from_config(path)


class VLLMChatbot(Chatbot):
    """
    vLLM offline inference chatbot wrapper with native batched generation.
    """

    def __init__(
        self,
        id: str,
        config: Mapping[str, Any] | str | Path,
    ):
        super().__init__(id=id)
        self.config = VLLMChatbotConfig.from_config(config)

        self.prompt_template = "{user_prompt}"
        self.stop_sequences = []
        if self.config.prompt_template_path is not None:
            with open(self.config.prompt_template_path, "r", encoding="utf-8") as f:
                prompt_template_cfg = yaml.safe_load(f)

            self.prompt_template = prompt_template_cfg.get("template", "{user_prompt}")
            self.stop_sequences = prompt_template_cfg.get("stop_sequences", [])

        llm_cls, sampling_params_cls = _load_vllm_classes()
        self.sampling_params_cls = sampling_params_cls
        self.llm = llm_cls(model=self.config.model_path, **self.config.llm_kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, list[str]]:
        return self.generate_batch(
            prompts=[prompt],
            max_new_tokens=max_new_tokens,
        )[0]

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: Optional[int] = None,
    ) -> list[tuple[str, list[str]]]:
        templated_prompts = [
            self.prompt_template.format(user_prompt=prompt) for prompt in prompts
        ]
        sampling_params = self.sampling_params_cls(
            temperature=self.config.model_temperature,
            top_p=self.config.top_p,
            max_tokens=max_new_tokens or self.config.max_tokens,
            stop=self.stop_sequences or None,
        )

        outputs = self.llm.generate(
            templated_prompts,
            sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
            )

        generations = []
        for index, output in enumerate(outputs):
            if not output.outputs:
                raise RuntimeError(f"vLLM output {index} did not contain completions")
            generations.append((output.outputs[0].text, []))
        return generations


def _load_vllm_classes():
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams
