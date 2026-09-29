import csv
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from chatbots.base import Chatbot
from chatbots.factory import create_chatbot
from retrievers.factory import create_retriever
from utils.config import ChatbotSpec, ExperimentConfig, RagConfig
from utils.rag import (
    RESULT_COLUMNS,
    build_non_rag_prompts,
    build_rag_prompts,
    load_eval_data_csv,
)


RESULT_FILENAME = "results.csv"
CompletionKey = tuple[str, str]
CompletionCounts = Counter[CompletionKey]
PendingRow = tuple[dict[str, str], int]


def run_experiment(config: ExperimentConfig, k: int = 1) -> Path:
    if k < 1:
        raise ValueError("`k` must be a positive integer")

    eval_data_path = Path(config.eval_data.path).expanduser()
    eval_rows, eval_fieldnames = load_eval_data_csv(
        eval_data_path=eval_data_path,
        question_column=config.eval_data.question_column,
    )

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / RESULT_FILENAME
    fieldnames = [*eval_fieldnames, *RESULT_COLUMNS]
    expected_keys = {
        (eval_row[config.eval_data.question_column], chatbot_id)
        for eval_row in eval_rows
        for chatbot_spec in config.chatbots
        for chatbot_id in _chatbot_ids(
            chatbot_spec=chatbot_spec,
            rag_config=config.rag_config,
            include_non_rag=config.include_non_rag,
        )
    }
    non_rag_chatbot_ids = (
        {chatbot.id for chatbot in config.chatbots} if config.include_non_rag else set()
    )
    completed_counts = _prepare_results_for_resume(
        result_path=result_path,
        fieldnames=fieldnames,
        question_column=config.eval_data.question_column,
        expected_keys=expected_keys,
        non_rag_chatbot_ids=non_rag_chatbot_ids,
    )
    progress_total = sum(
        _pending_count(
            eval_rows=eval_rows,
            chatbot_id=chatbot_id,
            question_column=config.eval_data.question_column,
            k=k,
            completed_counts=completed_counts,
        )
        for chatbot_spec in config.chatbots
        for chatbot_id in _chatbot_ids(
            chatbot_spec=chatbot_spec,
            rag_config=config.rag_config,
            include_non_rag=config.include_non_rag,
        )
    )
    if progress_total == 0:
        return result_path

    has_header = result_path.exists() and result_path.stat().st_size > 0
    mode = "a" if has_header else "w"
    with result_path.open(
        mode,
        encoding="utf-8",
        newline="",
        buffering=1,
    ) as result_file:
        writer = csv.DictWriter(result_file, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()

        with tqdm(
            total=progress_total,
            desc="Generating responses",
            unit="call",
        ) as progress:
            for chatbot_spec in config.chatbots:
                _run_chatbot(
                    chatbot_spec=chatbot_spec,
                    eval_rows=eval_rows,
                    rag_config=config.rag_config,
                    question_column=config.eval_data.question_column,
                    include_non_rag=config.include_non_rag,
                    writer=writer,
                    k=k,
                    progress=progress,
                    completed_counts=completed_counts,
                )

    return result_path


def _prepare_results_for_resume(
    result_path: Path,
    fieldnames: list[str],
    question_column: str,
    expected_keys: set[CompletionKey],
    non_rag_chatbot_ids: set[str],
) -> CompletionCounts:
    if not result_path.exists() or result_path.stat().st_size == 0:
        return Counter()

    csv.field_size_limit(sys.maxsize)
    with result_path.open("r", encoding="utf-8", newline="") as result_file:
        reader = csv.DictReader(result_file)
        if reader.fieldnames != fieldnames:
            raise ValueError(
                f"Cannot resume experiment because results columns do not match: "
                f"{result_path}"
            )

        retained_rows = []
        completed_counts: CompletionCounts = Counter()
        retry_count = 0
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[field] is None for field in fieldnames):
                raise ValueError(
                    f"Cannot resume experiment from malformed results row "
                    f"{line_number}: {result_path}"
                )

            key = (row[question_column], row["chatbot_id"])
            has_stale_non_rag_prompt = (
                row["chatbot_id"] in non_rag_chatbot_ids
                and row["input_prompt"] != row[question_column]
            )
            if key in expected_keys and (
                row["error"].strip() or has_stale_non_rag_prompt
            ):
                retry_count += 1
                continue

            retained_rows.append(row)
            completed_counts[key] += 1

    if retry_count:
        _replace_results(result_path, fieldnames, retained_rows)
    return completed_counts


def _replace_results(
    result_path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=result_path.parent,
            prefix=f".{result_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, result_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _chatbot_ids(
    chatbot_spec: ChatbotSpec,
    rag_config: RagConfig,
    include_non_rag: bool,
) -> list[str]:
    chatbot_ids = [
        f"{chatbot_spec.id} ({retriever.id})" for retriever in rag_config.retrievers
    ]
    if include_non_rag:
        chatbot_ids.append(chatbot_spec.id)
    return chatbot_ids


def _pending_rows(
    eval_rows: list[dict[str, str]],
    chatbot_id: str,
    question_column: str,
    k: int,
    completed_counts: CompletionCounts,
) -> list[PendingRow]:
    used_counts: CompletionCounts = Counter()
    pending_rows = []
    for eval_row in eval_rows:
        key = (eval_row[question_column], chatbot_id)
        completed = min(k, max(0, completed_counts[key] - used_counts[key]))
        used_counts[key] += completed
        missing = k - completed
        if missing:
            pending_rows.append((eval_row, missing))
    return pending_rows


def _pending_count(
    eval_rows: list[dict[str, str]],
    chatbot_id: str,
    question_column: str,
    k: int,
    completed_counts: CompletionCounts,
) -> int:
    return sum(
        missing
        for _, missing in _pending_rows(
            eval_rows=eval_rows,
            chatbot_id=chatbot_id,
            question_column=question_column,
            k=k,
            completed_counts=completed_counts,
        )
    )


def _run_chatbot(
    chatbot_spec: ChatbotSpec,
    eval_rows: list[dict[str, str]],
    rag_config: RagConfig,
    question_column: str,
    include_non_rag: bool,
    writer: csv.DictWriter,
    k: int,
    progress: tqdm,
    completed_counts: CompletionCounts,
) -> None:
    pending_by_id = {
        chatbot_id: _pending_rows(
            eval_rows=eval_rows,
            chatbot_id=chatbot_id,
            question_column=question_column,
            k=k,
            completed_counts=completed_counts,
        )
        for chatbot_id in _chatbot_ids(
            chatbot_spec=chatbot_spec,
            rag_config=rag_config,
            include_non_rag=include_non_rag,
        )
    }
    if not any(pending_by_id.values()):
        return

    try:
        chatbot = create_chatbot(chatbot_spec)
    except Exception as exc:
        for chatbot_id, pending_rows in pending_by_id.items():
            _write_error_rows(
                chatbot_id=chatbot_id,
                pending_rows=pending_rows,
                prompts=[row[question_column] for row, _ in pending_rows],
                error=f"Failed to initialize chatbot: {exc}",
                writer=writer,
                progress=progress,
            )
        return

    for retriever_spec in rag_config.retrievers:
        chatbot_id = f"{chatbot_spec.id} ({retriever_spec.id})"
        pending_rows = pending_by_id[chatbot_id]
        if not pending_rows:
            continue

        prompt_rows = [row for row, _ in pending_rows]
        try:
            retriever = create_retriever(retriever_spec, chatbot)
            prompts = build_rag_prompts(
                eval_rows=prompt_rows,
                retriever=retriever,
                rag_config=rag_config,
                question_column=question_column,
            )
        except Exception as exc:
            _write_error_rows(
                chatbot_id=chatbot_id,
                pending_rows=pending_rows,
                prompts=[row[question_column] for row in prompt_rows],
                error=f"Failed to retrieve context: {exc}",
                writer=writer,
                progress=progress,
            )
            continue

        _run_prompt_set(
            chatbot=chatbot,
            chatbot_id=chatbot_id,
            batch_size=chatbot_spec.batch_size,
            pending_rows=pending_rows,
            prompts=prompts,
            writer=writer,
            progress=progress,
        )

    if include_non_rag:
        pending_rows = pending_by_id[chatbot_spec.id]
        if pending_rows:
            prompt_rows = [row for row, _ in pending_rows]
            _run_prompt_set(
                chatbot=chatbot,
                chatbot_id=chatbot_spec.id,
                batch_size=chatbot_spec.batch_size,
                pending_rows=pending_rows,
                prompts=build_non_rag_prompts(
                    eval_rows=prompt_rows,
                    question_column=question_column,
                ),
                writer=writer,
                progress=progress,
            )


def _run_prompt_set(
    chatbot: Chatbot,
    chatbot_id: str,
    batch_size: int,
    pending_rows: list[PendingRow],
    prompts: list[str],
    writer: csv.DictWriter,
    progress: tqdm,
) -> None:
    work_items = [
        (eval_row, prompt)
        for (eval_row, repetitions), prompt in zip(
            pending_rows,
            prompts,
            strict=True,
        )
        for _ in range(repetitions)
    ]
    for start in range(0, len(work_items), batch_size):
        chunk = work_items[start : start + batch_size]
        chunk_prompts = [prompt for _, prompt in chunk]

        try:
            generations = list(
                chatbot.generate_batch(prompts=chunk_prompts, max_new_tokens=None)
            )
            if len(generations) != len(chunk):
                raise RuntimeError(
                    "Batch generation returned "
                    f"{len(generations)} results for {len(chunk)} prompts"
                )
        except Exception:
            for eval_row, prompt in chunk:
                try:
                    generation = chatbot.generate(prompt=prompt, max_new_tokens=None)
                    response = _response_from_generation(generation)
                    error = ""
                except Exception as exc:
                    response = ""
                    error = str(exc)
                _write_result_row(
                    chatbot_id=chatbot_id,
                    eval_row=eval_row,
                    prompt=prompt,
                    response=response,
                    error=error,
                    writer=writer,
                    progress=progress,
                )
            continue

        for (eval_row, prompt), generation in zip(chunk, generations, strict=True):
            _write_result_row(
                chatbot_id=chatbot_id,
                eval_row=eval_row,
                prompt=prompt,
                response=_response_from_generation(generation),
                error="",
                writer=writer,
                progress=progress,
            )


def _response_from_generation(generation: object) -> str:
    return generation[0] if isinstance(generation, tuple) else str(generation)


def _write_result_row(
    chatbot_id: str,
    eval_row: dict[str, str],
    prompt: str,
    response: str,
    error: str,
    writer: csv.DictWriter,
    progress: tqdm,
) -> None:
    writer.writerow(
        {
            **eval_row,
            "chatbot_id": chatbot_id,
            "input_prompt": prompt,
            "response": response,
            "error": error,
        }
    )
    progress.update()


def _write_error_rows(
    chatbot_id: str,
    pending_rows: list[PendingRow],
    prompts: list[str],
    error: str,
    writer: csv.DictWriter,
    progress: tqdm,
) -> None:
    for (eval_row, repetitions), prompt in zip(
        pending_rows,
        prompts,
        strict=True,
    ):
        for _ in range(repetitions):
            _write_result_row(
                chatbot_id=chatbot_id,
                eval_row=eval_row,
                prompt=prompt,
                response="",
                error=error,
                writer=writer,
                progress=progress,
            )
