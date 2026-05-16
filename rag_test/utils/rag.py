import csv
from pathlib import Path
from typing import Protocol

import yaml

from utils.config import RagConfig
from utils.embeddings import encode_texts


RESULT_COLUMNS = ["chatbot_id", "input_prompt", "response", "error"]


class Retriever(Protocol):
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, str]]:
        pass


def load_eval_data_csv(
    eval_data_path: Path,
    question_column: str,
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Loads eval data csv with checks for missing values
    """
    if not eval_data_path.exists():
        raise FileNotFoundError(f"Eval data CSV not found: {eval_data_path}")

    try:
        with eval_data_path.open("r", encoding="utf-8", newline="") as eval_data_file:
            reader = csv.DictReader(eval_data_file, strict=True)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"Eval data CSV is empty: {eval_data_path}")
            if question_column not in fieldnames:
                raise ValueError(
                    f"Eval data CSV must contain a `{question_column}` column: "
                    f"{eval_data_path}"
                )

            overlapping_columns = set(fieldnames) & set(RESULT_COLUMNS)
            if overlapping_columns:
                raise ValueError(
                    f"Eval data CSV cannot contain reserved result columns: "
                    f"{sorted(overlapping_columns)}"
                )

            rows = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Malformed CSV at {eval_data_path}: row {line_number} has "
                        "more fields than the header"
                    )
                if row[question_column] is None:
                    raise ValueError(
                        f"Malformed CSV at {eval_data_path}: row {line_number} is "
                        f"missing a `{question_column}` value"
                    )
                rows.append(row)
            return rows, fieldnames
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV at {eval_data_path}: {exc}") from exc


def build_rag_prompts(
    eval_rows: list[dict[str, str]],
    retriever: Retriever,
    rag_config: RagConfig,
    question_column: str,
) -> list[str]:
    """
    Builds prompts from config/templates
    """
    prompt_template = _load_template(Path(rag_config.prompt_template_path))
    retrieved_item_template = _load_template(
        Path(rag_config.retrieved_item_template_path)
    )
    eval_embeddings = encode_texts(
        texts=[row[question_column] for row in eval_rows],
        embedding_model=rag_config.embedding_model,
    )

    prompts = []
    for eval_row, eval_embedding in zip(
        eval_rows,
        eval_embeddings,
        strict=True,
    ):
        retrieved_rows = retriever.search(
            query_embedding=eval_embedding,
            top_k=rag_config.top_k,
        )
        retrieved_context = "\n".join(
            _format_template(retrieved_item_template, row) for row in retrieved_rows
        )
        prompts.append(
            _format_template(
                prompt_template,
                {
                    "query": eval_row[question_column],
                    "retrieved_context": retrieved_context,
                },
            )
        )
    return prompts


def build_non_rag_prompts(
    eval_rows: list[dict[str, str]],
    rag_config: RagConfig,
    question_column: str,
) -> list[str]:
    """
    Builds baseline prompts from the RAG prompt template with no retrieved context.
    """
    prompt_template = _load_template(Path(rag_config.prompt_template_path))
    return [
        _format_template(
            prompt_template,
            {
                "query": eval_row[question_column],
                "retrieved_context": "",
            },
        )
        for eval_row in eval_rows
    ]


def _load_template(path: Path) -> str:
    """
    Loads templates with checks for improper formatting
    """
    if not path.exists():
        raise FileNotFoundError(f"Template file not found: {path}")

    with path.open("r", encoding="utf-8") as template_file:
        template_config = yaml.safe_load(template_file)

    if not isinstance(template_config, dict):
        raise ValueError(f"Template file must contain a mapping: {path}")

    template = template_config.get("template")
    if not isinstance(template, str) or not template:
        raise ValueError(f"Template file must contain a non-empty `template`: {path}")
    return template


def _format_template(template: str, values: dict[str, str]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"Template references missing field `{exc.args[0]}`") from exc
