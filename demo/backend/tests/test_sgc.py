import asyncio
import unittest
from unittest.mock import Mock, patch

import requests
import httpx

from src.api import main as api_main
from src.api.chatbots.sgc import SGCChatbot
from src.api.warmup import WarmupStatus


class SGCChatbotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chatbot = SGCChatbot(id="SGC", config_path="configs/chatbots/sgc.yaml")

    @patch("src.api.chatbots.sgc.requests.post")
    def test_sends_prompt_to_solve_endpoint(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {"solution": "SGC answer", "system": "System1"}
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
        self.assertEqual(response.system, "System2")
        self.assertEqual(response.sources, [])
        coordinator.start.assert_called_once()
        self.assertEqual(post.call_args.args, ("http://127.0.0.1:8001/solve",))

    @patch("src.api.chatbots.sgc.requests.post")
    def test_api_preserves_solver_per_response_and_handles_missing_solver(self, post: Mock):
        coordinator = Mock()
        coordinator.start.return_value = WarmupStatus(status="ready")
        bot = SGCChatbot(id="SGC", config_path="configs/chatbots/sgc.yaml")

        async def check_responses():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
            ) as client:
                for system in ("System1", "System2", None, "unknown"):
                    with self.subTest(system=system):
                        payload = {"solution": "Answer"}
                        if system is not None:
                            payload["system"] = system
                        post.return_value.json.return_value = payload
                        response = await client.post(
                            "/generate", json={"prompt": "Question", "model": "SGC"}
                        )
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json()["text"], "Answer")
                        expected = system if system in ("System1", "System2") else None
                        self.assertEqual(response.json()["system"], expected)

                # An error must not carry the previous request's solver.
                post.side_effect = requests.Timeout
                response = await client.post(
                    "/generate", json={"prompt": "Question", "model": "SGC"}
                )
                self.assertEqual(response.json()["text"], "SGC request timed out")
                self.assertIsNone(response.json()["system"])

        with (
            patch.dict(api_main.chatbots, {"SGC": bot}, clear=True),
            patch.object(api_main, "warmup_coordinator", coordinator),
        ):
            asyncio.run(check_responses())


if __name__ == "__main__":
    unittest.main()
