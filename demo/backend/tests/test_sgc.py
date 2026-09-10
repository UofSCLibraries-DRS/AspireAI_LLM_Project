import unittest
from unittest.mock import Mock, patch

import requests

from src.api import main as api_main
from src.api.chatbots.sgc import SGCChatbot


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
            timeout=30.0,
        )
        response.raise_for_status.assert_called_once_with()

    @patch("src.api.chatbots.sgc.requests.post", side_effect=requests.Timeout)
    def test_reports_timeout(self, _: Mock) -> None:
        self.assertEqual(self.chatbot.generate("Question"), ("SGC request timed out", []))

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


if __name__ == "__main__":
    unittest.main()
