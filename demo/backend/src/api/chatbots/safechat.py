import re
from dataclasses import dataclass

import requests
import yaml

from .base import Chatbot


@dataclass
class SafeChatConfig:
    url: str
    sender: str
    timeout: int

    @classmethod
    def from_yaml(cls, path: str) -> "SafeChatConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise TypeError("YAML file must contain a mapping at the top level")

        required_fields = {f.name for f in cls.__dataclass_fields__.values()}
        missing = required_fields - data.keys()

        if missing:
            raise ValueError(f"Missing required config fields in `{path}`: {missing}")

        return cls(**data)


class SafeChat(Chatbot):
    """
    SafeChat wrapper.
    """

    def __init__(
        self,
        id: str,
        config_path: str,
    ):
        super().__init__(id=id)

        # Load config
        self.config = SafeChatConfig.from_yaml(config_path)

    def _extract_sources(self, text: str) -> tuple[str, list[str]]:
        """Extract source URLs from [Source: ...; Date: ] pattern and remove it from text."""
        match = re.search(r"\[Source:\s*(.*?);\s*Date:.*?\]", text)
        if not match:
            return text, []

        sources_str = match.group(1)
        sources = [s.strip() for s in sources_str.split(",") if s.strip()]
        cleaned_text = re.sub(r"\[Source:.*?;\s*Date:.*?\]\s*", "", text).strip()
        return cleaned_text, sources

    def _format_bullets(self, text: str) -> str:
        """Convert inline + bullet points to newline-separated list."""
        # Split on " +" but keep the + by using a lookahead
        parts = re.split(r"\s+(?=\+)", text)
        return "\n".join(part.strip() for part in parts)

    def generate(self, prompt: str, max_new_tokens=None) -> tuple[str, list[str]]:
        try:
            response = requests.post(
                f"{self.config.url}/webhooks/rest/webhook",
                headers={"Content-Type": "application/json"},
                json={
                    "sender": self.config.sender,
                    "message": prompt,
                },
                timeout=self.config.timeout,
            )

            if response.status_code != 200:
                return f"SafeChat error: HTTP {response.status_code}"

            replies = response.json()

            if not replies:
                return "SafeChat returned an empty response"

            text = replies[0].get("text", "No response received")

            text, sources = self._extract_sources(text)

            text = self._format_bullets(text)

            return text, sources

        except requests.exceptions.Timeout:
            return "SafeChat request timed out", []

        except requests.exceptions.ConnectionError:
            return "Could not connect to SafeChat service", []

        except Exception as e:
            return f"SafeChat error: {e!s}", []
