import asyncio
import unittest
from unittest.mock import Mock, patch

import requests

from src.api import main as api_main
from src.api.chatbots.sgc import SGCChatbot
from src.api.warmup import WarmupStatus


class SGCChatbotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chatbot = SGCChatbot(id="SGC", config_path="configs/chatbots/sgc.yaml")

    @patch("src.api.chatbots.sgc.requests.post")
    def test_sends_prompt_to_solve_endpoint(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"solution": "SGC answer", "system": "system_1"}
        post.return_value = response

        answer, sources = self.chatbot.generate("What happened?", max_new_tokens=12)

        self.assertEqual((answer, sources), ("SGC answer", []))
        post.assert_called_once_with(
            "http://127.0.0.1:8001/solve",
            json={"problem": "What happened?"},
            timeout=300.0,
        )
        response.raise_for_status.assert_called_once_with()

    @patch("src.api.chatbots.sgc.requests.post", side_effect=requests.Timeout)
    def test_reports_timeout(self, _: Mock) -> None:
        self.assertEqual(self.chatbot.generate("Question"), ("SGC request timed out", []))

    @patch("src.api.chatbots.sgc.requests.post")
    def test_reports_upstream_http_status(self, post: Mock) -> None:
        response = requests.Response()
        response.status_code = 502
        post.return_value = response

        self.assertEqual(self.chatbot.generate("Question"), ("SGC error: HTTP 502", []))

    @patch("src.api.chatbots.sgc.requests.post")
    def test_rejects_response_without_solution(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"system": "system_1"}
        post.return_value = response

        self.assertEqual(
            self.chatbot.generate("Question"),
            ("SGC returned a response without a solution", []),
        )


class SGCAPIIntegrationTests(unittest.TestCase):
    def test_generate_request_accepts_sgc_model(self) -> None:
        request = api_main.GenerateRequest(prompt="question", model="SGC")
        self.assertEqual(request.model, "SGC")

    @patch("src.api.chatbots.sgc.requests.post")
    def test_generate_routes_sgc_after_m14_is_ready(self, post: Mock) -> None:
        post.return_value.json.return_value = {
            "solution": "Selected answer", "system": "System2"
        }
        coordinator = Mock()
        coordinator.start.return_value = WarmupStatus(status="ready")
        bot = SGCChatbot(id="SGC", config_path="configs/chatbots/sgc.yaml")
        with (
            patch.dict(api_main.chatbots, {"SGC": bot}, clear=True),
            patch.object(api_main, "warmup_coordinator", coordinator),
        ):
            response = asyncio.run(api_main.generate(
                api_main.GenerateRequest(prompt="Question", model="SGC")
            ))
        self.assertEqual(response.model, "SGC")
        self.assertEqual(response.text, "Selected answer")
        self.assertEqual(response.sources, [])
        coordinator.start.assert_called_once()
        self.assertEqual(post.call_args.args, ("http://127.0.0.1:8001/solve",))


if __name__ == "__main__":
    unittest.main()
