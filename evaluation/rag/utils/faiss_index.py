import csv
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from utils.embeddings import EMBEDDING_COLUMN


BUILD_CHUNK_SIZE = 1024


@dataclass(frozen=True)
class FaissIndexPaths:
    index_path: Path
    metadata_path: Path
    offsets_path: Path

    @classmethod
    def from_embedding_cache_path(cls, embedding_cache_path: Path) -> "FaissIndexPaths":
        return cls(
            index_path=embedding_cache_path.with_suffix(".faiss"),
            metadata_path=embedding_cache_path.with_suffix(".metadata.jsonl"),
            offsets_path=embedding_cache_path.with_suffix(".offsets.npy"),
        )

    def all_exist(self) -> bool:
        return (
            self.index_path.exists()
            and self.metadata_path.exists()
            and self.offsets_path.exists()
        )


class FaissRetriever:
    def __init__(
        self,
        paths: FaissIndexPaths,
        index,
        offsets: np.ndarray,
    ):
        if index.ntotal != len(offsets):
            raise ValueError(
                "FAISS index and metadata offset counts do not match: "
                f"{index.ntotal} != {len(offsets)}"
            )

        self.paths = paths
        self.index = index
        self.offsets = offsets

    @classmethod
    def load(cls, paths: FaissIndexPaths) -> "FaissRetriever":
        return cls(
            paths=paths,
            index=faiss.read_index(str(paths.index_path)),
            offsets=np.load(paths.offsets_path),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, str]]:
        if top_k < 1:
            return []

        query_matrix = np.asarray([query_embedding], dtype=np.float32)
        _normalize_rows(query_matrix)
        _, indices = self.index.search(query_matrix, top_k)

        rows = []
        with self.paths.metadata_path.open("rb") as metadata_file:
            for index_id in indices[0]:
                if index_id < 0:
                    continue
                metadata_file.seek(int(self.offsets[index_id]))
                rows.append(json.loads(metadata_file.readline().decode("utf-8")))
        return rows


def ensure_faiss_index(embedding_cache_path: Path) -> FaissRetriever:
    paths = FaissIndexPaths.from_embedding_cache_path(embedding_cache_path)
    if paths.all_exist():
        print(f"FAISS index already exists: {paths.index_path}")
        return FaissRetriever.load(paths)

    build_faiss_index(embedding_cache_path=embedding_cache_path, paths=paths)
    print(f"Saved FAISS index: {paths.index_path}")
    return FaissRetriever.load(paths)


def build_faiss_index(
    embedding_cache_path: Path,
    paths: FaissIndexPaths,
) -> None:
    """
    Builds index from csv containing embeddings
    """
    if not embedding_cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {embedding_cache_path}")

    index = None
    offsets = []
    embedding_chunk = []

    try:
        with embedding_cache_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as embedding_file:
            reader = csv.DictReader(embedding_file, strict=True)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"Embedding cache is empty: {embedding_cache_path}")
            if EMBEDDING_COLUMN not in fieldnames:
                raise ValueError(
                    f"Embedding cache must contain an `{EMBEDDING_COLUMN}` column: "
                    f"{embedding_cache_path}"
                )

            with paths.metadata_path.open("wb") as metadata_file:
                for line_number, row in enumerate(reader, start=2):
                    if None in row:
                        raise ValueError(
                            f"Malformed embedding cache at {embedding_cache_path}: "
                            f"row {line_number} has more fields than the header"
                        )

                    embedding = _parse_embedding(
                        row.pop(EMBEDDING_COLUMN),
                        embedding_cache_path,
                        line_number,
                    )
                    if index is None:
                        index = faiss.IndexFlatIP(len(embedding))

                    embedding_chunk.append(embedding)
                    offsets.append(metadata_file.tell())
                    metadata_file.write(
                        json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
                    )

                    if len(embedding_chunk) >= BUILD_CHUNK_SIZE:
                        _add_chunk(index, embedding_chunk)
                        embedding_chunk = []

        if index is None:
            raise ValueError(f"Embedding cache contains no rows: {embedding_cache_path}")

        if embedding_chunk:
            _add_chunk(index, embedding_chunk)

        faiss.write_index(index, str(paths.index_path))
        np.save(paths.offsets_path, np.asarray(offsets, dtype=np.int64))
    except csv.Error as exc:
        raise ValueError(f"Malformed embedding cache at {embedding_cache_path}: {exc}") from exc


def _add_chunk(index, embedding_chunk: list[list[float]]) -> None:
    matrix = np.asarray(embedding_chunk, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embedding chunk must be a two-dimensional matrix")
    if matrix.shape[1] != index.d:
        raise ValueError(
            "Embedding dimensions do not match FAISS index dimension: "
            f"{matrix.shape[1]} != {index.d}"
        )

    _normalize_rows(matrix)
    index.add(matrix)


def _normalize_rows(matrix: np.ndarray) -> None:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    nonzero_norms = norms[:, 0] > 0.0
    matrix[nonzero_norms] = matrix[nonzero_norms] / norms[nonzero_norms]


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
    return [float(value) for value in embedding]
