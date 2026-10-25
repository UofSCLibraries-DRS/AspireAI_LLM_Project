import json
import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import boto3
import yaml
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from .base import Chatbot, ChatbotSource

logger = logging.getLogger(__name__)


class BedrockModelNotReadyError(RuntimeError):
    """An imported model needs restoration before inference can proceed."""


@dataclass
class BedrockChatbotConfig:
    model_id: str
    model_temperature: float
    prompt_template_path: str
    max_tokens: int = 512
    top_p: float = 0.9
    region_name: str = "us-east-1"

    @classmethod
    def from_yaml(cls, path: str) -> "BedrockChatbotConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise TypeError("YAML file must contain a mapping at the top level")

        required_fields = {"model_id", "model_temperature", "prompt_template_path"}
        missing = required_fields - data.keys()

        if missing:
            raise ValueError(f"Missing required config fields in `{path}`: {missing}")

        return cls(**data)


class BedrockChatbot(Chatbot):
    """
    AWS Bedrock chatbot wrapper using InvokeModel API.
    Supports base and instruction-tuned models with explicit prompt templates.
    """

    def __init__(
        self,
        id: str,
        config_path: str,
    ):
        super().__init__(id=id)
        self.on_success: Callable[[], None] | None = None

        # Read config
        self.config = BedrockChatbotConfig.from_yaml(config_path)

        # Get stop sequences and template from template config
        with open(self.config.prompt_template_path, "r") as f:
            prompt_template_cfg = yaml.safe_load(f)

        self.prompt_template = prompt_template_cfg.get(
            "template", "{user_prompt}"
        )  # Default to identity function
        self.stop_sequences = prompt_template_cfg.get("stop_sequences", [])

        # Initialize Bedrock client
        # AWS credentials should be loaded from .env at top level
        # or from ~/.aws/credentials
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
        text = text[:min_idx].strip()
        return text

    def generate(
        self, prompt: str, max_new_tokens: int | None = None
    ) -> tuple[str, list[ChatbotSource]]:
        """
        Generate a response using AWS Bedrock InvokeModel API.

        Args:
            prompt: The user's prompt
            max_new_tokens: Maximum number of tokens to generate (uses config default if None)

        Returns:
            Generated text response
        """
        try:
            text = self.invoke(prompt, max_new_tokens)
            if self.on_success is not None:
                try:
                    self.on_success()
                except Exception:
                    # Coordination failure must not discard a successful answer.
                    logger.exception("Unable to record Bedrock inference readiness")
            return text, []

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "ModelNotReadyException":
                raise BedrockModelNotReadyError("Model is starting") from e
            error_message = e.response.get("Error", {}).get("Message", str(e))
            return f"Bedrock API error [{error_code}]: {error_message}", []

        except KeyError as e:
            return f"Bedrock response parsing error: missing key {e!s}", []

        except json.JSONDecodeError as e:
            return f"Bedrock JSON parsing error: {e!s}", []

        except Exception as e:  # noqa: BLE001
            return f"Bedrock error: {e!s}", []

    def warmup_client(self, remaining_seconds: float):
        """A dedicated, single-attempt client; the coordinator owns retries."""
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=self.config.region_name,
            config=BotoConfig(
                connect_timeout=min(5, remaining_seconds),
                read_timeout=min(310, remaining_seconds),
                retries={"total_max_attempts": 1, "mode": "standard"},
            ),
        )

    def generate_with_confidence(
        self, prompt: str, max_new_tokens: int | None = None
    ) -> tuple[str, float]:
        """Return text and geometric-mean output-token probability for SGC."""
        try:
            text, payload = self._invoke(prompt, max_new_tokens, return_logprobs=True)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") == "ModelNotReadyException":
                raise BedrockModelNotReadyError("Model is starting") from error
            raise

        if self.on_success is not None:
            try:
                self.on_success()
            except Exception:
                logger.exception("Unable to record Bedrock inference readiness")

        logprobs = payload.get("logprobs")
        if not isinstance(logprobs, list) or not logprobs:
            raise ValueError("Bedrock did not return output-token log probabilities")
        values = []
        for entry in logprobs:
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError("Invalid Bedrock output-token log probabilities")
            value = next(iter(entry.values()))
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value > 0
            ):
                raise ValueError("Invalid Bedrock output-token log probability")
            values.append(value)
        return text, math.exp(statistics.fmean(values))

    def invoke(
        self, prompt: str, max_new_tokens: int | None = None, *, client=None
    ) -> str:
        """Invoke and validate a response, propagating errors to the caller."""
        text, _ = self._invoke(prompt, max_new_tokens, client=client)
        return text

    def _invoke(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        *,
        client=None,
        return_logprobs: bool = False,
    ) -> tuple[str, dict]:
        templated_prompt = self.prompt_template.format(user_prompt=prompt)
        request_body = {
            "prompt": templated_prompt,
            "max_gen_len": self.config.max_tokens
            if max_new_tokens is None
            else max_new_tokens,
            "temperature": self.config.model_temperature,
            "top_p": self.config.top_p,
        }
        if return_logprobs:
            request_body["return_logprobs"] = True
        response = (self.client if client is None else client).invoke_model(
            modelId=self.config.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        body = response["body"]
        try:
            response_body = json.loads(body.read())
        finally:
            body.close()
        if not isinstance(response_body, dict) or not isinstance(
            response_body.get("generation"), str
        ):
            raise TypeError("Bedrock response must contain a generation string")
        text = self._post_process(response_body["generation"], templated_prompt)
        return text, response_body
