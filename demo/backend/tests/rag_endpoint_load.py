"""Manually measure response times for any live chatbot mode.

This is a standalone load-test script, not an automated unit test. Start the
backend first, then run, for example:

    uv run python tests/rag_endpoint_load.py

The script performs one baseline request followed by three 16-request
scenarios: one burst, two starts per second, and four starts per second.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_URL = "http://localhost:8000/generate"
DEFAULT_PROMPT = "Who was John H. McCray?"
REQUEST_COUNT = 16
AVAILABLE_MODES = ("M8", "M9", "LLAMA", "RAG", "SC")


@dataclass(frozen=True)
class RequestResult:
    number: int
    scheduled_start: float
    actual_start: float
    elapsed: float
    status_code: int | None
    response_characters: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def query_endpoint(
    *,
    number: int,
    scenario_started: float,
    scheduled_start: float,
    url: str,
    mode: str,
    prompt: str,
    max_new_tokens: int,
    timeout: float,
) -> RequestResult:
    """Send one request and capture client-observed time to the full response."""
    started = time.monotonic()
    status_code: int | None = None
    response_characters = 0

    try:
        response = requests.post(
            url,
            json={
                "prompt": prompt,
                "model": mode,
                "max_new_tokens": max_new_tokens,
            },
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        status_code = response.status_code
        response.raise_for_status()

        body: Any = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("text"), str):
            raise ValueError("response JSON does not contain a string 'text' field")
        response_characters = len(body["text"])
        error = None
    except (requests.RequestException, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    finished = time.monotonic()
    return RequestResult(
        number=number,
        scheduled_start=scheduled_start,
        actual_start=started - scenario_started,
        elapsed=finished - started,
        status_code=status_code,
        response_characters=response_characters,
        error=error,
    )


def percentile(values: list[float], percentage: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def print_results(
    name: str,
    results: list[RequestResult],
    wall_time: float,
) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for result in sorted(results, key=lambda item: item.number):
        status = (
            f"HTTP {result.status_code}"
            if result.status_code is not None
            else "no HTTP response"
        )
        line = (
            f"Request {result.number:02d} | scheduled +{result.scheduled_start:5.2f}s "
            f"| started +{result.actual_start:5.2f}s | {status} "
            f"| response {result.elapsed:7.2f}s"
        )
        if result.succeeded:
            line += f" | {result.response_characters} chars"
        else:
            line += f" | ERROR: {result.error}"
        print(line)

    successful_times = [result.elapsed for result in results if result.succeeded]
    failures = len(results) - len(successful_times)
    print(
        f"Completed {len(results)} requests in {wall_time:.2f}s "
        f"({len(successful_times)} succeeded, {failures} failed)."
    )
    if successful_times:
        print(
            "Response time summary: "
            f"min {min(successful_times):.2f}s | "
            f"median {statistics.median(successful_times):.2f}s | "
            f"mean {statistics.fmean(successful_times):.2f}s | "
            f"p95 {percentile(successful_times, 0.95):.2f}s | "
            f"max {max(successful_times):.2f}s"
        )


def run_scenario(
    *,
    name: str,
    request_count: int,
    starts_per_second: int | None,
    url: str,
    mode: str,
    prompt: str,
    max_new_tokens: int,
    timeout: float,
) -> list[RequestResult]:
    """Run a burst or start fixed-size batches at one-second intervals."""
    print(f"\nStarting {name}...", flush=True)
    scenario_started = time.monotonic()
    futures: list[Future[RequestResult]] = []

    with ThreadPoolExecutor(
        max_workers=request_count,
        thread_name_prefix="rag-load",
    ) as executor:
        for request_index in range(request_count):
            scheduled_start = (
                0.0
                if starts_per_second is None
                else float(request_index // starts_per_second)
            )
            delay = scenario_started + scheduled_start - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            futures.append(
                executor.submit(
                    query_endpoint,
                    number=request_index + 1,
                    scenario_started=scenario_started,
                    scheduled_start=scheduled_start,
                    url=url,
                    mode=mode,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    timeout=timeout,
                )
            )

        results = [future.result() for future in futures]

    wall_time = time.monotonic() - scenario_started
    print_results(name, results, wall_time)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure a live chatbot mode under four request patterns."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Full /generate endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--mode",
        choices=AVAILABLE_MODES,
        default="RAG",
        help="Backend mode to test (default: RAG)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt sent by every simulated user (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=128,
        help="Maximum generated tokens per response (default: 128)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=180.0,
        help="Per-request timeout in seconds (default: 180)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = (
        (f"1. {args.mode} single-query baseline", 1, None),
        (f"2. {args.mode}: sixteen queries started at once", REQUEST_COUNT, None),
        (
            f"3. {args.mode}: sixteen queries, two started every second",
            REQUEST_COUNT,
            2,
        ),
        (
            f"4. {args.mode}: sixteen queries, four started every second",
            REQUEST_COUNT,
            4,
        ),
    )

    any_failures = False
    try:
        for name, request_count, starts_per_second in scenarios:
            results = run_scenario(
                name=name,
                request_count=request_count,
                starts_per_second=starts_per_second,
                url=args.url,
                mode=args.mode,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                timeout=args.timeout,
            )
            any_failures = any_failures or any(
                not result.succeeded for result in results
            )

            if request_count == 1 and any_failures:
                print("\nBaseline request failed; stopping before the load scenarios.")
                return 1
    except KeyboardInterrupt:
        print("\nLoad test interrupted.")
        return 130

    return 1 if any_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
