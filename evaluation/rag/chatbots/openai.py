import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from openai import OpenAI

from .base import Chatbot
from .config import config_from_mapping_or_yaml


@dataclass
class OpenAIChatbotConfig:
    model_id: str
    base_url: str
    model_temperature: float = 0.5

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str | Path,
    ) -> "OpenAIChatbotConfig":
        return config_from_mapping_or_yaml(
            config_cls=cls,
            config=config,
            aliases={"temperature": "model_temperature"},
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpenAIChatbotConfig":
        return cls.from_config(path)


class OpenAIChatbot(Chatbot):
    def __init__(
        self,
        id: str,
        config: Mapping[str, Any] | str | Path,
    ):
        super().__init__(id=id)
        self.config = OpenAIChatbotConfig.from_config(config)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, list[str]]:
        request = {
            "model": self.config.model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.model_temperature,
        }
        if max_new_tokens is not None:
            request["max_tokens"] = max_new_tokens

        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content, []
