"""Host-wide imported-model readiness for multiple Linux API workers.

Job locks are never replaced: each background thread owns an open lock descriptor
until it finishes. A separate lock protects atomic status-file replacements.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import random
import tempfile
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Literal

from botocore.exceptions import (
    ClientError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
)
from botocore.exceptions import (
    ConnectionError as BotoConnectionError,
)
from pydantic import BaseModel

from .chatbots.bedrock import BedrockChatbot

logger = logging.getLogger("uvicorn.error.bedrock_warmup")
FRESH_SECONDS = 240
COOLDOWN_SECONDS = 30
RETRY_BUDGET_SECONDS = 600
FAILURE_MESSAGE = "The model could not start. Please try again shortly."


class WarmupStatus(BaseModel):
    model: Literal["M14"] = "M14"
    status: Literal["cold", "warming", "ready", "failed"]
    message: str | None = None


class WarmupCoordinator:
    def __init__(self, chatbot: BedrockChatbot, state_dir: str | Path | None = None):
        self.chatbot = chatbot
        self.directory = Path(
            state_dir
            or os.environ.get(
                "BEDROCK_WARMUP_STATE_DIR",
                str(Path(tempfile.gettempdir()) / f"aspireai-bedrock-{os.getuid()}"),
            )
        )
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = hashlib.sha256(
            f"{chatbot.config.region_name}:{chatbot.config.model_id}".encode()
        ).hexdigest()
        self.state_path = self.directory / f"{key}.json"
        self.state_lock_path = self.directory / f"{key}.state.lock"
        self.job_lock_path = self.directory / f"{key}.job.lock"

    @staticmethod
    def _open_lock(path: Path):
        return os.fdopen(os.open(path, os.O_CREAT | os.O_RDWR, 0o600), "r+")

    @contextmanager
    def _state_lock(self):
        with self._open_lock(self.state_lock_path) as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _try_job_lock(self):
        lock = self._open_lock(self.job_lock_path)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return None
        except OSError:
            lock.close()
            raise
        return lock

    def _read(self) -> dict:
        try:
            data = json.loads(self.state_path.read_text())
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError):
            return {}

    def _write(self, data: dict) -> None:
        # Readers also take the state lock; replace prevents partial JSON after a crash.
        fd, name = tempfile.mkstemp(dir=self.directory, prefix=".warmup-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream)
            os.replace(name, self.state_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def _recent(timestamp, seconds: float) -> bool:
        return (
            isinstance(timestamp, (int, float))
            and 0 <= time.time() - timestamp < seconds
        )

    def _status(self, data: dict) -> WarmupStatus:
        if self._recent(data.get("last_success"), FRESH_SECONDS):
            return WarmupStatus(status="ready")
        lock = self._try_job_lock()
        if lock is None:
            return WarmupStatus(status="warming")
        lock.close()
        if self._recent(data.get("failed_at"), COOLDOWN_SECONDS):
            return WarmupStatus(status="failed", message=FAILURE_MESSAGE)
        # A recorded "warming" state without a held job lock means its worker died.
        return WarmupStatus(status="cold")

    def status(self) -> WarmupStatus:
        with self._state_lock():
            return self._status(self._read())

    def start(self) -> WarmupStatus:
        with self._state_lock():
            data = self._read()
            status = self._status(data)
            if status.status != "cold":
                return status
            lock = self._try_job_lock()
            if lock is None:
                return WarmupStatus(status="warming")
            try:
                self._write({"status": "warming", "started_at": time.time()})
                threading.Thread(
                    target=self._run, args=(lock,), daemon=True, name="bedrock-warmup"
                ).start()
            except Exception:
                lock.close()
                raise
            return WarmupStatus(status="warming")

    def mark_success(self) -> None:
        with self._state_lock():
            self._write({"status": "ready", "last_success": time.time()})

    def invalidate(self) -> None:
        with self._state_lock():
            self._write({"status": "cold"})

    @staticmethod
    def _retryable(error: Exception) -> bool:
        if isinstance(error, ClientError):
            return error.response.get("Error", {}).get("Code") in {
                "ModelNotReadyException",
                "ThrottlingException",
                "ServiceUnavailableException",
                "InternalServerException",
                "ModelTimeoutException",
            }
        return isinstance(
            error,
            (
                BotoConnectionError,
                ReadTimeoutError,
                IncompleteReadError,
                ResponseStreamingError,
            ),
        )

    def _run(self, lock) -> None:
        started = time.monotonic()
        deadline = started + RETRY_BUDGET_SECONDS
        attempts = 0
        logger.info("Bedrock warm-up started model=%s", self.chatbot.id)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Bedrock warm-up retry budget exhausted")
                attempts += 1
                try:
                    # Closing this client cannot affect normal generation's client.
                    with closing(self.chatbot.warmup_client(remaining)) as client:
                        self.chatbot.invoke("Hello", max_new_tokens=1, client=client)
                    self.mark_success()
                    logger.info(
                        "Bedrock warm-up ready attempts=%s duration=%.2fs",
                        attempts,
                        time.monotonic() - started,
                    )
                    return
                except Exception as error:
                    if not self._retryable(error):
                        raise
                    logger.info(
                        "Bedrock warm-up retry attempt=%s error=%s",
                        attempts,
                        type(error).__name__,
                    )
                    delay = min(15, 2 ** min(attempts - 1, 4)) * random.uniform(0.5, 1)
                    time.sleep(max(0, min(delay, deadline - time.monotonic())))
        except Exception:
            logger.exception(
                "Bedrock warm-up failed attempts=%s duration=%.2fs",
                attempts,
                time.monotonic() - started,
            )
            try:
                with self._state_lock():
                    data = self._read()
                    # A concurrent real inference may already have proved readiness.
                    if not self._recent(data.get("last_success"), FRESH_SECONDS):
                        self._write({"status": "failed", "failed_at": time.time()})
            except Exception:
                logger.exception("Unable to persist Bedrock warm-up failure")
        finally:
            lock.close()
