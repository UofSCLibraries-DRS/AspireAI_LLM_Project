import csv
import os
from pathlib import Path

from tqdm import tqdm

from chatbots.factory import create_chatbot
from utils.cache import get_embedding_cache_path
from utils.config import ChatbotSpec, ExperimentConfig
from utils.embeddings import precompute_embedding_cache
from utils.faiss_index import ensure_faiss_index
from utils.rag import (
    RESULT_COLUMNS,
    build_non_rag_prompts,
    build_rag_prompts,
    load_eval_data_csv,
)


RESULT_FILENAME = "results.csv"
DATA_FOLDER_ENV_VAR = "DATA_FOLDER"
RAG_ID_SUFFIX = " (RAG)"


def run_experiment(config: ExperimentConfig, k: int = 1) -> Path:
    if k < 1:
        raise ValueError("`k` must be a positive integer")

    data_path = _resolve_data_path(config.data)
    eval_data_path = _resolve_data_path(config.eval_data.path)

    cache_path = _ensure_embedding_cache(config, data_path)
    retriever = ensure_faiss_index(cache_path)
    eval_rows, eval_fieldnames = load_eval_data_csv(
        eval_data_path=eval_data_path,
        question_column=config.eval_data.question_column,
    )
    rag_prompts = build_rag_prompts(
        eval_rows=eval_rows,
        retriever=retriever,
        rag_config=config.rag_config,
        question_column=config.eval_data.question_column,
    )
    non_rag_prompts = (
        build_non_rag_prompts(
            eval_rows=eval_rows,
            rag_config=config.rag_config,
            question_column=config.eval_data.question_column,
        )
        if config.include_non_rag
        else None
    )

    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / RESULT_FILENAME

    with result_path.open("w", encoding="utf-8", newline="") as result_file:
        writer = csv.DictWriter(
            result_file,
            fieldnames=[*eval_fieldnames, *RESULT_COLUMNS],
        )
        writer.writeheader()

        prompt_set_count = 2 if config.include_non_rag else 1
        progress_total = len(config.chatbots) * len(eval_rows) * k * prompt_set_count
        with tqdm(
            total=progress_total,
            desc="Generating responses",
            unit="call",
        ) as progress:
            for chatbot_spec in config.chatbots:
                _run_chatbot(
                    chatbot_spec=chatbot_spec,
                    eval_rows=eval_rows,
                    rag_prompts=rag_prompts,
                    non_rag_prompts=non_rag_prompts,
                    writer=writer,
                    k=k,
                    progress=progress,
                )

    return result_path


def _ensure_embedding_cache(config: ExperimentConfig, data_path: Path) -> Path:
    """
    Checks for a pre-computed embedding based on the data and embedding model defined in experiment config
    """
    cache_path = get_embedding_cache_path(
        data=config.data,
        embedding_model=config.rag_config.embedding_model,
    )

    if cache_path.exists():
        print(f"Embedding cache already exists: {cache_path}")
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    precompute_embedding_cache(
        data_path=data_path,
        embedding_model=config.rag_config.embedding_model,
        cache_path=cache_path,
    )
    print(f"Saved embedding cache: {cache_path}")
    return cache_path


def _resolve_data_path(path: str) -> Path:
    configured_path = Path(path).expanduser()
    if configured_path.is_absolute():
        return configured_path

    data_folder = Path(os.getenv(DATA_FOLDER_ENV_VAR, ".")).expanduser()
    return data_folder / configured_path


def _run_chatbot(
    chatbot_spec: ChatbotSpec,
    eval_rows: list[dict[str, str]],
    rag_prompts: list[str],
    non_rag_prompts: list[str] | None,
    writer: csv.DictWriter,
    k: int,
    progress: tqdm,
) -> None:
    prompt_sets = [
        (f"{chatbot_spec.id}{RAG_ID_SUFFIX}", rag_prompts),
    ]
    if non_rag_prompts is not None:
        prompt_sets.append((chatbot_spec.id, non_rag_prompts))

    try:
        chatbot = create_chatbot(chatbot_spec)
    except Exception as exc:
        for chatbot_id, _prompts in prompt_sets:
            _write_error_rows(
                chatbot_id=chatbot_id,
                eval_rows=eval_rows,
                prompts=_prompts,
                error=f"Failed to initialize chatbot: {exc}",
                writer=writer,
                k=k,
                progress=progress,
            )
        return

    for chatbot_id, prompts in prompt_sets:
        _run_prompt_set(
            chatbot=chatbot,
            chatbot_id=chatbot_id,
            batch_size=chatbot_spec.batch_size,
            eval_rows=eval_rows,
            prompts=prompts,
            writer=writer,
            k=k,
            progress=progress,
        )


def _run_prompt_set(
    chatbot: object,
    chatbot_id: str,
    batch_size: int,
    eval_rows: list[dict[str, str]],
    prompts: list[str],
    writer: csv.DictWriter,
    k: int,
    progress: tqdm,
) -> None:
    work_items = [
        (eval_row, prompt)
        for eval_row, prompt in zip(eval_rows, prompts, strict=True)
        for _ in range(k)
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
    eval_rows: list[dict[str, str]],
    prompts: list[str],
    error: str,
    writer: csv.DictWriter,
    k: int,
    progress: tqdm,
) -> None:
    for eval_row, prompt in zip(eval_rows, prompts, strict=True):
        for _ in range(k):
            _write_result_row(
                chatbot_id=chatbot_id,
                eval_row=eval_row,
                prompt=prompt,
                response="",
                error=error,
                writer=writer,
                progress=progress,
            )
