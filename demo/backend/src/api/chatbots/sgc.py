"""Client adapter for the locally hosted SafeGenChat (SGC) service."""

from dataclasses import dataclass

import requests
import yaml

from .base import Chatbot, ChatbotSource
from .conversation import ConversationTurn, recent_context


@dataclass
class SGCChatbotConfig:
    """Connection settings for SafeGenChat's ``POST /solve`` endpoint."""

    url: str
    timeout: float = 300.0

    @classmethod
    def from_yaml(cls, path: str) -> "SGCChatbotConfig":
        with open(path) as config_file:
            data = yaml.safe_load(config_file)

        if not isinstance(data, dict):
            raise TypeError("YAML file must contain a mapping at the top level")

        required_fields = {"url"}
        missing = required_fields - data.keys()
        if missing:
            raise ValueError(f"Missing required config fields in `{path}`: {missing}")

        return cls(**data)


class SGCChatbot(Chatbot):
    """Send prompts to a SafeGenChat service running on this host."""

    def __init__(self, id: str, config_path: str):
        super().__init__(id=id)
        self.config = SGCChatbotConfig.from_yaml(config_path)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        *,
        context: list[ConversationTurn] | None = None,
    ) -> tuple[str, list[ChatbotSource]]:
        text, sources, _ = self.generate_with_system(
            prompt, max_new_tokens, context=context
        )
        return text, sources

    def generate_with_system(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        *,
        context: list[ConversationTurn] | None = None,
    ) -> tuple[str, list[ChatbotSource], str | None]:
        """Generate a solution through SGC.

        SGC controls its own decoding settings, so ``max_new_tokens`` is not
        part of its public API and is intentionally not forwarded.
        """
        del max_new_tokens
        request_payload: dict[str, object] = {"problem": prompt}
        history = recent_context(context)
        if history:
            request_payload["context"] = [turn.model_dump() for turn in history]
        try:
            response = requests.post(
                f"{self.config.url.rstrip('/')}/solve",
                json=request_payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "SGC request timed out", [], None
        except requests.exceptions.ConnectionError:
            return "Could not connect to the SGC service", [], None
        except requests.exceptions.HTTPError as error:
            status_code = (
                error.response.status_code if error.response is not None else "unknown"
            )
            return f"SGC error: HTTP {status_code}", [], None
        except requests.exceptions.RequestException as error:
            return f"SGC request failed: {error!s}", [], None
        except ValueError:
            return "SGC returned an invalid JSON response", [], None

        solution = payload.get("solution") if isinstance(payload, dict) else None
        if not isinstance(solution, str):
            return "SGC returned a response without a solution", [], None
        system = payload.get("system")
        if system not in ("System1", "System2"):
            system = None
        sources = payload.get("sources", []) if system == "System1" else []
        if not isinstance(sources, list):
            sources = []
        return solution, sources, system
