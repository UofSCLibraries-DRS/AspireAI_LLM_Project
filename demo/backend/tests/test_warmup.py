import asyncio
import io
import json
import multiprocessing
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import boto3
import httpx
from botocore.exceptions import ClientError, EndpointConnectionError
from botocore.response import StreamingBody
from botocore.stub import Stubber
from src.api import main as api_main
from src.api.chatbots.bedrock import BedrockChatbot, BedrockModelNotReadyError
from src.api.warmup import FRESH_SECONDS, WarmupCoordinator, WarmupStatus


def aws_error(code):
    return ClientError(
        {"Error": {"Code": code, "Message": "private AWS detail"}}, "InvokeModel"
    )


class BlockingChatbot:
    """Pickle/fork-friendly fake with process-shared inference counters."""

    id = "M14"
    config = SimpleNamespace(region_name="us-east-1", model_id="imported/test")

    def __init__(self, calls, entered, release):
        self.calls, self.entered, self.release = calls, entered, release

    def warmup_client(self, remaining):
        return SimpleNamespace(close=lambda: None)

    def invoke(self, *args, **kwargs):
        with self.calls.get_lock():
            self.calls.value += 1
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("test invocation was not released")
        return ""


def process_start(directory, calls, entered, release, gate, results):
    coordinator = WarmupCoordinator(BlockingChatbot(calls, entered, release), directory)
    gate.wait(10)
    results.put(coordinator.start().status)
    release.wait(10)
    for thread in threading.enumerate():
        if thread.name == "bedrock-warmup":
            thread.join(5)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bot = Mock(
            id="M14",
            config=SimpleNamespace(
                model_id="imported/test",
                region_name="us-east-1",
            ),
        )
        self.clients = []

        def make_client(remaining):
            # Match boto3: close() is supported, __enter__/__exit__ are not.
            client = Mock(spec=["close"])
            self.clients.append(client)
            return client

        self.bot.warmup_client.side_effect = make_client
        self.bot.invoke.return_value = ""
        self.coordinator = WarmupCoordinator(self.bot, self.temp.name)

    def finish(self, coordinator=None):
        coordinator = coordinator or self.coordinator
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            state = coordinator.status()
            if state.status != "warming":
                return state
            time.sleep(0.005)
        self.fail("background job did not finish")

    def test_get_is_read_only_and_success_expires(self):
        self.assertEqual(self.coordinator.status().status, "cold")
        self.bot.invoke.assert_not_called()
        self.coordinator.mark_success()
        stored = self.coordinator.state_path.read_text()
        self.assertEqual(self.coordinator.status().status, "ready")
        self.assertEqual(self.coordinator.state_path.read_text(), stored)
        with patch(
            "src.api.warmup.time.time", return_value=time.time() + FRESH_SECONDS + 1
        ):
            self.assertEqual(self.coordinator.status().status, "cold")

    def test_warmup_uses_one_token_and_normal_success_is_shared(self):
        self.assertEqual(self.coordinator.start().status, "warming")
        self.assertEqual(self.finish().status, "ready")
        self.bot.invoke.assert_called_once()
        self.assertEqual(self.bot.invoke.call_args.args, ("Hello",))
        self.assertEqual(self.bot.invoke.call_args.kwargs["max_new_tokens"], 1)
        self.assertIs(self.bot.invoke.call_args.kwargs["client"], self.clients[0])
        self.clients[0].close.assert_called_once_with()
        second = WarmupCoordinator(self.bot, self.temp.name)
        self.assertEqual(second.start().status, "ready")
        self.assertEqual(self.bot.invoke.call_count, 1)

    def test_retries_temporary_errors(self):
        self.bot.invoke.side_effect = [
            aws_error("ModelNotReadyException"),
            aws_error("ThrottlingException"),
            EndpointConnectionError(endpoint_url="https://bedrock.invalid"),
            "",
        ]
        with patch("src.api.warmup.time.sleep"):
            self.coordinator.start()
            # Join without relying on the mocked sleep to schedule the worker.
            for thread in threading.enumerate():
                if thread.name == "bedrock-warmup":
                    thread.join(3)
        self.assertEqual(self.coordinator.status().status, "ready")
        self.assertEqual(self.bot.invoke.call_count, 4)
        self.assertEqual(len(self.clients), 4)
        for client in self.clients:
            client.close.assert_called_once_with()

    def test_fatal_failure_has_cooldown_and_hides_details(self):
        self.bot.invoke.side_effect = aws_error("AccessDeniedException")
        self.coordinator.start()
        status = self.finish()
        self.assertEqual(status.status, "failed")
        self.assertNotIn("private", status.message)
        self.assertEqual(self.coordinator.start().status, "failed")
        self.assertEqual(self.bot.invoke.call_count, 1)
        self.clients[0].close.assert_called_once_with()
        self.bot.invoke.side_effect = None
        with patch("src.api.warmup.time.time", return_value=time.time() + 31):
            self.assertEqual(self.coordinator.start().status, "warming")
            self.assertEqual(self.finish().status, "ready")

    def test_retry_budget_exhaustion(self):
        self.bot.invoke.side_effect = aws_error("ModelNotReadyException")
        with patch("src.api.warmup.RETRY_BUDGET_SECONDS", 0.01):
            self.coordinator.start()
            self.assertEqual(self.finish().status, "failed")
        self.assertEqual(self.bot.invoke.call_count, 1)

    def test_malformed_response_never_marks_ready(self):
        self.bot.invoke.side_effect = ValueError("missing generation")
        self.coordinator.start()
        self.assertEqual(self.finish().status, "failed")
        self.assertEqual(self.bot.invoke.call_count, 1)

    def test_model_and_region_isolate_state(self):
        self.coordinator.mark_success()
        for attribute, value in [
            ("model_id", "other-model"),
            ("region_name", "us-west-2"),
        ]:
            with self.subTest(attribute=attribute):
                bot = Mock(config=SimpleNamespace(**vars(self.bot.config)))
                setattr(bot.config, attribute, value)
                other = WarmupCoordinator(bot, self.temp.name)
                self.assertEqual(other.status().status, "cold")

    def test_corrupt_status_can_be_recovered(self):
        self.coordinator.state_path.write_text("{")
        self.assertEqual(self.coordinator.status().status, "cold")
        self.coordinator.start()
        self.assertEqual(self.finish().status, "ready")

    def test_job_lock_inode_is_preserved(self):
        self.coordinator.status()
        inode = self.coordinator.job_lock_path.stat().st_ino
        self.coordinator.start()
        self.finish()
        self.coordinator.mark_success()
        self.assertEqual(self.coordinator.job_lock_path.stat().st_ino, inode)

    def test_two_processes_start_only_one_inference(self):
        ctx = multiprocessing.get_context("fork")
        calls, entered, release, gate, results = (
            ctx.Value("i", 0),
            ctx.Event(),
            ctx.Event(),
            ctx.Event(),
            ctx.Queue(),
        )
        processes = [
            ctx.Process(
                target=process_start,
                args=(
                    self.temp.name,
                    calls,
                    entered,
                    release,
                    gate,
                    results,
                ),
            )
            for _ in range(2)
        ]
        try:
            for process in processes:
                process.start()
            gate.set()
            self.assertTrue(entered.wait(5))
            self.assertEqual(
                [results.get(timeout=5) for _ in processes], ["warming"] * 2
            )
            self.assertEqual(calls.value, 1)
            self.assertEqual(self.coordinator.status().status, "warming")
        finally:
            release.set()
            for process in processes:
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
        self.assertEqual([p.exitcode for p in processes], [0, 0])
        self.assertEqual(self.coordinator.status().status, "ready")

    def test_worker_death_releases_lock_and_allows_restart(self):
        ctx = multiprocessing.get_context("fork")
        calls, entered, release, gate, results = (
            ctx.Value("i", 0),
            ctx.Event(),
            ctx.Event(),
            ctx.Event(),
            ctx.Queue(),
        )
        process = ctx.Process(
            target=process_start,
            args=(
                self.temp.name,
                calls,
                entered,
                release,
                gate,
                results,
            ),
        )
        process.start()
        try:
            gate.set()
            self.assertTrue(entered.wait(5))
            self.assertEqual(results.get(timeout=5), "warming")
            process.kill()
            process.join(5)
            self.assertEqual(self.coordinator.status().status, "cold")
            self.assertEqual(self.coordinator.start().status, "warming")
            self.assertEqual(self.finish().status, "ready")
        finally:
            if process.is_alive():
                process.kill()
            process.join(5)


class BedrockWarmupTests(unittest.TestCase):
    @patch("src.api.chatbots.bedrock.boto3.client")
    def setUp(self, client):
        self.client = client.return_value
        self.bot = BedrockChatbot("M14", "configs/chatbots/m14_bedrock.yaml")

    def test_warmup_client_has_no_sdk_retries_and_bounded_timeout(self):
        with patch("src.api.chatbots.bedrock.boto3.client") as factory:
            self.bot.warmup_client(20)
            config = factory.call_args.kwargs["config"]
            self.assertEqual(config.retries["total_max_attempts"], 1)
            self.assertEqual(config.read_timeout, 20)
            self.assertEqual(config.connect_timeout, 5)
            self.bot.warmup_client(600)
            self.assertEqual(factory.call_args.kwargs["config"].read_timeout, 310)

    def test_warmup_with_real_sdk_client_and_stubbed_inference(self):
        # Exercise the actual SDK client lifecycle without credentials or AWS calls.
        client = boto3.session.Session().client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        payload = b'{"generation": ""}'
        expected = {
            "modelId": self.bot.config.model_id,
            "body": json.dumps(
                {
                    "prompt": self.bot.prompt_template.format(user_prompt="Hello"),
                    "max_gen_len": 1,
                    "temperature": self.bot.config.model_temperature,
                    "top_p": self.bot.config.top_p,
                }
            ),
            "contentType": "application/json",
            "accept": "application/json",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            Stubber(client) as stubber,
            patch.object(self.bot, "warmup_client", return_value=client),
            patch.object(client, "close", wraps=client.close) as close,
        ):
            stubber.add_response(
                "invoke_model",
                {
                    "body": StreamingBody(io.BytesIO(payload), len(payload)),
                    "contentType": "application/json",
                },
                expected,
            )
            coordinator = WarmupCoordinator(self.bot, directory)
            coordinator._run(coordinator._try_job_lock())
            self.assertEqual(coordinator.status().status, "ready")
            close.assert_called_once_with()
            stubber.assert_no_pending_responses()

    def test_invoke_raises_on_malformed_body_and_closes_it(self):
        body = io.BytesIO(b'{"error": "not a generation"}')
        self.client.invoke_model.return_value = {"body": body}
        with self.assertRaises(TypeError):
            self.bot.invoke("Hello", 1)
        self.assertTrue(body.closed)

    def test_only_successful_normal_generation_refreshes_readiness(self):
        self.bot.on_success = Mock()
        self.client.invoke_model.return_value = {
            "body": io.BytesIO(b'{"generation": "Answer"}')
        }
        self.assertEqual(self.bot.generate("Question"), ("Answer", []))
        self.bot.on_success.assert_called_once()
        self.bot.on_success.reset_mock()
        self.client.invoke_model.side_effect = aws_error("AccessDeniedException")
        self.bot.generate("Question")
        self.bot.on_success.assert_not_called()

    def test_model_not_ready_is_typed_and_not_answer_text(self):
        self.client.invoke_model.side_effect = aws_error("ModelNotReadyException")
        with self.assertRaises(BedrockModelNotReadyError):
            self.bot.generate("Question")


class WarmupAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.coordinator = Mock()
        self.coordinator.start.return_value = WarmupStatus(status="warming")
        self.coordinator.status.return_value = WarmupStatus(status="cold")
        self.patch_coordinator = patch.object(
            api_main, "warmup_coordinator", self.coordinator
        )
        self.patch_coordinator.start()
        self.addCleanup(self.patch_coordinator.stop)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api_main.app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)

    async def test_endpoint_status_codes_and_get_does_not_start(self):
        response = await self.client.get("/warmup")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cold")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.coordinator.start.assert_not_called()
        self.assertEqual((await self.client.post("/warmup")).status_code, 202)
        self.coordinator.start.return_value = WarmupStatus(status="ready")
        self.assertEqual((await self.client.post("/warmup")).status_code, 200)
        self.coordinator.start.return_value = WarmupStatus(
            status="failed", message="Try again"
        )
        self.assertEqual((await self.client.post("/warmup")).status_code, 503)

    async def test_uninitialized_coordinator_is_safe_unavailable(self):
        with patch.object(api_main, "warmup_coordinator", None):
            for method in (self.client.get, self.client.post):
                response = await method("/warmup")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["status"], "failed")
                self.assertEqual(response.headers["Cache-Control"], "no-store")

    async def test_m14_and_rag_share_readiness_gate(self):
        bot = Mock()
        with patch.dict(api_main.chatbots, {"M14": bot, "RAG": bot}, clear=True):
            for model in ("M14", "RAG"):
                response = await self.client.post(
                    "/generate", json={"model": model, "prompt": "Question"}
                )
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()["detail"]["code"], "model_warming")
            bot.generate.assert_not_called()
            self.assertEqual(self.coordinator.start.call_count, 2)

    async def test_loss_of_readiness_restarts_warmup(self):
        self.coordinator.start.return_value = WarmupStatus(status="ready")
        bot = Mock()
        bot.generate.side_effect = BedrockModelNotReadyError("Starting")
        with patch.dict(api_main.chatbots, {"M14": bot}, clear=True):
            response = await self.client.post(
                "/generate", json={"model": "M14", "prompt": "Question"}
            )
        self.coordinator.invalidate.assert_called_once()
        self.assertEqual(self.coordinator.start.call_count, 2)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "model_warming")

    async def test_state_directory_failure_is_safe_unavailable(self):
        self.coordinator.start.side_effect = PermissionError("private file path")
        self.coordinator.status.side_effect = PermissionError("private file path")
        with patch.dict(api_main.chatbots, {"M14": Mock()}, clear=True):
            responses = [
                await self.client.get("/warmup"),
                await self.client.post("/warmup"),
                await self.client.post(
                    "/generate", json={"model": "M14", "prompt": "Question"}
                ),
            ]
        for response in responses:
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertNotIn("private", response.text)

    async def test_other_models_bypass_gate_and_inference_does_not_block_status(self):
        entered, release = threading.Event(), threading.Event()

        def blocking_generate(**kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test request was not released")
            return "Answer", []

        bot = Mock()
        bot.generate.side_effect = blocking_generate
        with patch.dict(api_main.chatbots, {"SC": bot}, clear=True):
            request = asyncio.create_task(
                self.client.post(
                    "/generate", json={"model": "SC", "prompt": "Question"}
                )
            )
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                status = await asyncio.wait_for(self.client.get("/warmup"), timeout=1)
                self.assertEqual(status.status_code, 200)
                self.coordinator.start.assert_not_called()
            finally:
                release.set()
            response = await request
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Answer")


if __name__ == "__main__":
    unittest.main()
