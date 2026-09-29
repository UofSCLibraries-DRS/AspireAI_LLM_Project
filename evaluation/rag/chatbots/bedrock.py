import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import boto3
import yaml
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .base import Chatbot
from .config import config_from_mapping_or_yaml


@dataclass
class BedrockCompletionChatbotConfig:
    model_id: str
    prompt_template_path: str
    model_temperature: float = 0.5
    max_tokens: int = 512
    top_p: float = 0.9
    region_name: str = "us-east-1"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | str | Path,
    ) -> "BedrockCompletionChatbotConfig":
        return config_from_mapping_or_yaml(
            config_cls=cls,
            config=config,
            aliases={
                "temperature": "model_temperature",
                "chat_template_path": "prompt_template_path",
            },
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BedrockCompletionChatbotConfig":
        return cls.from_config(path)


class BedrockCompletionChatbot(Chatbot):
    """
    AWS Bedrock chatbot wrapper using InvokeModel API.
    """

    def __init__(
        self,
        id: str,
        config: Mapping[str, Any] | str | Path,
    ):
        super().__init__(id=id)
        self.config = BedrockCompletionChatbotConfig.from_config(config)

        with open(self.config.prompt_template_path, "r", encoding="utf-8") as f:
            prompt_template_cfg = yaml.safe_load(f)

        self.prompt_template = prompt_template_cfg.get("template", "{user_prompt}")
        self.stop_sequences = prompt_template_cfg.get("stop_sequences", [])

        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.config.region_name,
            config=BotoConfig(
                read_timeout=150,
                retries={
                    "total_max_attempts": 10,
                    "mode": "standard",
                },
            ),
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
        max_new_tokens: Optional[int] = None,
    ) -> tuple[str, list[str]]:
        if max_new_tokens is None:
            max_new_tokens = self.config.max_tokens

        try:
            templated_prompt = self.prompt_template.format(user_prompt=prompt)
            request_body = {
                "prompt": templated_prompt,
                "max_gen_len": max_new_tokens,
                "temperature": self.config.model_temperature,
                "top_p": self.config.top_p,
            }

            response = self.client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(request_body),
            )
            response_body = json.loads(response["body"].read())
            return self._post_process(response_body["generation"], templated_prompt), []
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "Unknown")
            error_message = exc.response.get("Error", {}).get("Message", str(exc))
            raise RuntimeError(
                f"Bedrock API error [{error_code}]: {error_message}"
            ) from exc
        except KeyError as exc:
            raise RuntimeError(
                f"Bedrock response parsing error: missing key {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Bedrock JSON parsing error: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Bedrock error: {exc}") from exc
