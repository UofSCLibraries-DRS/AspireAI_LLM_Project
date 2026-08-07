import os
from dataclasses import dataclass

import yaml
from openai import OpenAI

from .base import Chatbot


@dataclass
class OpenAIChatbotConfig:
    model_id: str
    model_temperature: float
    base_url: str  # OpenAI client can work with other URLs

    @classmethod
    def from_yaml(cls, path: str) -> "OpenAIChatbotConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise TypeError("YAML file must contain a mapping at the top level")

        required_fields = {f.name for f in cls.__dataclass_fields__.values()}
        missing = required_fields - data.keys()

        if missing:
            raise ValueError(f"Missing required config fields in `{path}`: {missing}")

        return cls(**data)


class OpenAIChatbot(Chatbot):
    def __init__(
        self,
        id: str,
        config_path: str,
    ):
        super().__init__(id=id)

        # Read config
        self.config = OpenAIChatbotConfig.from_yaml(config_path)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
        )

    def generate(
        self, prompt: str, max_new_tokens: int | None
    ) -> tuple[str, list[str]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=self.config.model_temperature,
        )

        return response.choices[0].message.content, []
