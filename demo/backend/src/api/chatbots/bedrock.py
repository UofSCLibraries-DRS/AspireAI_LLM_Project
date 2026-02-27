from typing import Optional
from dataclasses import dataclass
import json
import yaml
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig

from .base import Chatbot

# TODO:
#   Test cold start issues...
#   How long are cold starts taking, is it feasible to wait for the cold start in this code?


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
            raise ValueError("YAML file must contain a mapping at the top level")

        required_fields = {"model_id", "model_temperature", "prompt_template_path"}
        missing = required_fields - data.keys()

        if missing:
            raise ValueError(f"Missing required config fields in `{path}`: {missing}")

        return cls(**data)


class BedrockChatbot(Chatbot):
    """
    AWS Bedrock chatbot wrapper using InvokeModel API.
    Designed for base models and fine-tuned variants with custom prompt templates.
    """

    def __init__(
        self,
        id: str,
        config_path: str,
    ):
        super().__init__(id=id)

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
        self, prompt: str, max_new_tokens: Optional[int] = None
    ) -> (str, list[str]):
        """
        Generate a response using AWS Bedrock InvokeModel API.

        Args:
            prompt: The user's prompt
            max_new_tokens: Maximum number of tokens to generate (uses config default if None)

        Returns:
            Generated text response
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_tokens

        try:
            # Apply prompt template
            templated_prompt = self.prompt_template.format(user_prompt=prompt)

            # Prepare request body for Llama models
            # Format based on Meta Llama documentation
            request_body = {
                "prompt": templated_prompt,
                "max_gen_len": max_new_tokens,
                "temperature": self.config.model_temperature,
                "top_p": self.config.top_p,
            }

            # Call Bedrock InvokeModel API
            response = self.client.invoke_model(
                modelId=self.config.model_id,
                body=json.dumps(request_body),
            )

            # Parse response
            response_body = json.loads(response["body"].read())

            # Extract generated text
            # Llama models return: {"generation": "text", "prompt_token_count": N, ...}

            return self._post_process(response_body["generation"], templated_prompt), []

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            return f"Bedrock API error [{error_code}]: {error_message}", []

        except KeyError as e:
            return f"Bedrock response parsing error: missing key {str(e)}", []

        except json.JSONDecodeError as e:
            return f"Bedrock JSON parsing error: {str(e)}", []

        except Exception as e:
            return f"Bedrock error: {str(e)}", []
