from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .base import Chatbot
from .config import config_from_mapping_or_yaml


@dataclass
class DummyChatbotConfig:
    prefix: str = "Dummy response"
    preview_chars: int = 180

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str | Path,
    ) -> "DummyChatbotConfig":
        return config_from_mapping_or_yaml(config_cls=cls, config=config)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DummyChatbotConfig":
        return cls.from_config(path)


class DummyChatbot(Chatbot):
    def __init__(
        self,
        id: str,
        config: Mapping[str, Any] | str | Path,
    ):
        super().__init__(id=id)
        self.config = DummyChatbotConfig.from_config(config)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, list[str]]:
        normalized_prompt = " ".join(prompt.split())
        preview = normalized_prompt[: self.config.preview_chars]
        return f"{self.config.prefix} from {self.id}: {preview}", []
