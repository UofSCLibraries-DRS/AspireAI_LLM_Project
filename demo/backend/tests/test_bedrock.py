import asyncio
import io
import json
import unittest
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from pydantic import ValidationError
from src.api import main as api_main
from src.api.chatbots.bedrock import BedrockChatbot
from src.api.warmup import WarmupStatus

########################################################################
###                                                                  ###
### Run from backend root with `uv run python -m tests.test_bedrock` ###
###                                                                  ###
########################################################################

# Load AWS credentials from .env file
# Must be done BEFORE creating the BedrockChatbot
load_dotenv()


class BedrockChatbotTests(unittest.TestCase):
    @patch("src.api.chatbots.bedrock.boto3.client")
    def setUp(self, client: Mock) -> None:
        self.client = client.return_value
        self.chatbot = BedrockChatbot(
            id="M14", config_path="configs/chatbots/m14_bedrock.yaml"
        )

    def respond_with(self, generation: str) -> None:
        self.client.invoke_model.return_value = {
            "body": io.BytesIO(json.dumps({"generation": generation}).encode())
        }

    def test_m14_sends_instruct_prompt_to_imported_model(self) -> None:
        self.respond_with("A concise answer.<|eot_id|>Unwanted continuation")
        prompt = 'Who was McCray? Return {"name": "..."}.'

        self.assertEqual(
            self.chatbot.generate(prompt, max_new_tokens=64), ("A concise answer.", [])
        )

        kwargs = self.client.invoke_model.call_args.kwargs
        self.assertEqual(
            kwargs["modelId"],
            "arn:aws:bedrock:us-east-1:777384741382:imported-model/jyve0zcux4qi",
        )
        body = json.loads(kwargs["body"])
        self.assertEqual(body["max_gen_len"], 64)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(
            body["prompt"],
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_m14_trims_echoed_prompt_and_control_tokens(self) -> None:
        templated = self.chatbot.prompt_template.format(user_prompt="Question")
        for marker in (
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|start_header_id|>user<|end_header_id|>",
        ):
            with self.subTest(marker=marker):
                self.respond_with(f"{templated} Answer {marker}More text")
                self.assertEqual(self.chatbot.generate("Question"), ("Answer", []))
                body = json.loads(self.client.invoke_model.call_args.kwargs["body"])
                self.assertEqual(body["max_gen_len"], 512)

    @patch("src.api.chatbots.bedrock.boto3.client")
    def test_original_m9_configuration_still_works(self, _: Mock) -> None:
        chatbot = BedrockChatbot(id="M9", config_path="configs/chatbots/m9_bedrock.yaml")
        self.assertEqual(
            chatbot.config.model_id,
            "arn:aws:bedrock:us-east-1:777384741382:imported-model/1a7xtypu9p6v",
        )
        self.assertIn(
            "<question>Question</question>",
            chatbot.prompt_template.format(user_prompt="Question"),
        )
        self.assertEqual(chatbot._post_process("Answer</answer>More text", ""), "Answer")

    def test_api_accepts_m14(self) -> None:
        coordinator = Mock()
        coordinator.start.return_value = WarmupStatus(status="ready")
        with (
            patch.dict(api_main.chatbots, {"M14": self.chatbot}, clear=True),
            patch.object(api_main, "warmup_coordinator", coordinator),
        ):
            request = api_main.GenerateRequest(prompt="Question", model="M14")
            self.respond_with("Answer<|eot_id|>")
            response = asyncio.run(api_main.generate(request))
            self.assertEqual(response.model, "M14")
            self.assertEqual(response.text, "Answer")

    def test_api_rejects_removed_m9_option(self) -> None:
        with self.assertRaises(ValidationError):
            api_main.GenerateRequest(prompt="Question", model="M9")


def main():
    # Path to M14 config
    config_path = "configs/chatbots/m14_bedrock.yaml"

    # Initialize the chatbot
    print("Initializing M14 Bedrock Chatbot...")
    m14_bot = BedrockChatbot(id="M14", config_path=config_path)

    print(f"Model ID: {m14_bot.config.model_id}")
    print(f"Temperature: {m14_bot.config.model_temperature}")
    print(f"Prompt Template: {m14_bot.prompt_template}")
    print(f"Stop Sequences: {m14_bot.stop_sequences}")
    print("-" * 50)

    # Test query
    test_prompt = "Hello, how are you today?"
    print(f"\nUser: {test_prompt}")

    # Generate response
    response = m14_bot.generate(test_prompt)
    print(f"Bot: {response}")
    print("-" * 50)

    # Optional: Test another query
    test_prompt_2 = "What organizations were John H. McCray affiliated with?"
    print(f"\nUser: {test_prompt_2}")
    response_2 = m14_bot.generate(test_prompt_2)
    print(f"Bot: {response_2}")


if __name__ == "__main__":
    main()
