from dataclasses import dataclass
import requests
import yaml
from .base import Chatbot

# TODO:
#   Refactor to just use the same process.
#   Should be able to use rasa.core.agent


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
            raise ValueError("YAML file must contain a mapping at the top level")

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

    def generate(self, prompt: str, max_new_tokens=None) -> str:
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

            return replies[0].get("text", "No response received")

        except requests.exceptions.Timeout:
            return "SafeChat request timed out"

        except requests.exceptions.ConnectionError:
            return "Could not connect to SafeChat service"

        except Exception as e:
            return f"SafeChat error: {str(e)}"
