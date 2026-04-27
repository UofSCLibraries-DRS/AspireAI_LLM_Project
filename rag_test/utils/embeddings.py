import csv
import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer


EMBEDDING_COLUMN = "embedding"
TEXT_COLUMN = "text"


def precompute_embedding_cache(
    data_path: Path,
    embedding_model: str,
    cache_path: Path,
) -> Path:
    rows, fieldnames = _read_data_csv(data_path)
    embeddings = encode_texts(
        texts=[row[TEXT_COLUMN] for row in rows],
        embedding_model=embedding_model,
    )
    _write_embedding_cache(
        cache_path=cache_path,
        rows=rows,
        fieldnames=fieldnames,
        embeddings=embeddings,
    )
    return cache_path


def load_embedding_cache(
    cache_path: Path,
) -> tuple[list[dict[str, str]], list[list[float]], list[str]]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")

    try:
        with cache_path.open("r", encoding="utf-8", newline="") as cache_file:
            reader = csv.DictReader(cache_file, strict=True)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"Embedding cache is empty: {cache_path}")
            if EMBEDDING_COLUMN not in fieldnames:
                raise ValueError(
                    f"Embedding cache must contain an `{EMBEDDING_COLUMN}` column: "
                    f"{cache_path}"
                )

            rows = []
            embeddings = []
            data_fieldnames = [
                fieldname for fieldname in fieldnames if fieldname != EMBEDDING_COLUMN
            ]
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Malformed embedding cache at {cache_path}: row {line_number} "
                        "has more fields than the header"
                    )
                embedding_json = row.pop(EMBEDDING_COLUMN)
                rows.append(row)
                embeddings.append(_parse_embedding(embedding_json, cache_path, line_number))
            return rows, embeddings, data_fieldnames
    except csv.Error as exc:
        raise ValueError(f"Malformed embedding cache at {cache_path}: {exc}") from exc


def _read_data_csv(data_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_path}")

    try:
        with data_path.open("r", encoding="utf-8", newline="") as data_file:
            reader = csv.DictReader(data_file, strict=True)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"Data CSV is empty: {data_path}")
            if TEXT_COLUMN not in fieldnames:
                raise ValueError(f"Data CSV must contain a `{TEXT_COLUMN}` column: {data_path}")
            if EMBEDDING_COLUMN in fieldnames:
                raise ValueError(
                    f"Data CSV already contains an `{EMBEDDING_COLUMN}` column: {data_path}"
                )
            rows = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Malformed CSV at {data_path}: row {line_number} has more "
                        "fields than the header"
                    )
                if row[TEXT_COLUMN] is None:
                    raise ValueError(
                        f"Malformed CSV at {data_path}: row {line_number} is missing "
                        f"a `{TEXT_COLUMN}` value"
                    )
                rows.append(row)
            return rows, fieldnames
    except csv.Error as exc:
        raise ValueError(f"Malformed CSV at {data_path}: {exc}") from exc


def encode_texts(texts: list[str], embedding_model: str) -> list[list[float]]:
    if not texts:
        return []

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts, show_progress_bar=True)
    return [_to_float_list(embedding) for embedding in embeddings]


def _to_float_list(embedding: Any) -> list[float]:
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def _parse_embedding(
    embedding_json: str,
    cache_path: Path,
    line_number: int,
) -> list[float]:
    try:
        embedding = json.loads(embedding_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed embedding JSON in {cache_path} row {line_number}: {exc}"
        ) from exc

    if not isinstance(embedding, list):
        raise ValueError(
            f"Embedding in {cache_path} row {line_number} must be a JSON list"
        )
    return _to_float_list(embedding)


def _write_embedding_cache(
    cache_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    embeddings: list[list[float]],
) -> None:
    cache_fieldnames = [*fieldnames, EMBEDDING_COLUMN]

    with cache_path.open("w", encoding="utf-8", newline="") as cache_file:
        writer = csv.DictWriter(cache_file, fieldnames=cache_fieldnames)
        writer.writeheader()

        for row, embedding in zip(rows, embeddings, strict=True):
            output_row = {
                **row,
                EMBEDDING_COLUMN: json.dumps(embedding, separators=(",", ":")),
            }
            writer.writerow(output_row)
