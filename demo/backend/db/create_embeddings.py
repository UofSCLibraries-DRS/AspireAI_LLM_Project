import argparse
import csv
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "intfloat/e5-base-v2"
PASSAGE_PREFIX = "passage: "
EMBEDDING_SIZE = 768
SOURCE_COLUMNS = ("title", "description", "transcript")
OUTPUT_COLUMNS = (
    "collection",
    "title",
    "description",
    "transcript",
    "title_embedding",
    "description_embedding",
    "transcript_embedding",
)

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "data" / "in" / "metadata"
DEFAULT_OUTPUT_PATH = SCRIPT_PATH.parent / "data" / "embeddings.csv"


def _raise_csv_field_limit() -> None:
    """Allow the csv module to read the largest transcripts in this dataset."""

    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


@dataclass(frozen=True)
class MetadataRow:
    collection: str
    title: str
    description: str
    transcript: str


class TextEmbedder(Protocol):
    """The subset of embedding behavior used by the CSV pipeline."""

    embedding_size: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Return one embedding per text and ``None`` for blank text."""


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` and reject unavailable explicitly requested devices."""

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested, but the MPS backend is unavailable")
    return device


class E5Embedder:
    """Embed complete texts with E5, aggregating all model-sized chunks."""

    embedding_size = EMBEDDING_SIZE

    def __init__(self, *, device: torch.device, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)

        hidden_size = int(self.model.config.hidden_size)
        if hidden_size != self.embedding_size:
            raise ValueError(
                f"Expected {self.embedding_size}-dimensional embeddings from "
                f"{MODEL_NAME}, but the model reports {hidden_size}"
            )

        tokenizer_limit = int(self.tokenizer.model_max_length)
        model_limit = int(self.model.config.max_position_embeddings)
        self.max_length = min(tokenizer_limit, model_limit)
        self.prefix_ids = self.tokenizer.encode(
            PASSAGE_PREFIX, add_special_tokens=False
        )
        if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
            raise ValueError(f"{MODEL_NAME} tokenizer is missing BERT special tokens")
        special_token_count = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.max_content_tokens = (
            self.max_length - special_token_count - len(self.prefix_ids)
        )
        if self.max_content_tokens < 1:
            raise ValueError("The model leaves no token capacity for source text")

        self.model.to(self.device)
        self.model.eval()

    def _token_chunks(self, text: str) -> Iterator[list[int]]:
        # Tokenize the complete value before splitting it; suppress the tokenizer's
        # pre-inference length warning because no oversized sequence reaches BERT.
        content_ids = self.tokenizer.encode(
            text, add_special_tokens=False, verbose=False
        )
        for start in range(0, len(content_ids), self.max_content_tokens):
            content_chunk = content_ids[start : start + self.max_content_tokens]
            yield [
                self.tokenizer.cls_token_id,
                *self.prefix_ids,
                *content_chunk,
                self.tokenizer.sep_token_id,
            ]

    @staticmethod
    def _average_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask[..., None].bool()
        masked_state = last_hidden_state.masked_fill(~mask, 0.0)
        return masked_state.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float] | None]:
        if not texts:
            return []

        chunks: list[tuple[int, list[int]]] = []
        has_text = torch.zeros(len(texts), dtype=torch.bool)
        for text_index, text in enumerate(texts):
            if not text.strip():
                continue
            has_text[text_index] = True
            chunks.extend(
                (text_index, input_ids) for input_ids in self._token_chunks(text)
            )

        # Group similarly sized chunks to avoid spending most inference work on padding.
        chunks.sort(key=lambda item: len(item[1]))
        sums = torch.zeros((len(texts), self.embedding_size), dtype=torch.float32)
        chunk_counts = torch.zeros(len(texts), dtype=torch.int64)

        with torch.inference_mode():
            for start in range(0, len(chunks), self.batch_size):
                batch = chunks[start : start + self.batch_size]
                owners = [owner for owner, _ in batch]
                tokenized = self.tokenizer.pad(
                    {"input_ids": [input_ids for _, input_ids in batch]},
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                model_inputs = {
                    key: value.to(self.device)
                    for key, value in tokenized.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
                outputs = self.model(**model_inputs)
                chunk_embeddings = self._average_pool(
                    outputs.last_hidden_state, model_inputs["attention_mask"]
                )
                chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
                chunk_embeddings = chunk_embeddings.to(
                    device="cpu", dtype=torch.float32
                )

                owner_tensor = torch.tensor(owners, dtype=torch.int64)
                sums.index_add_(0, owner_tensor, chunk_embeddings)
                chunk_counts.index_add_(
                    0, owner_tensor, torch.ones(len(owners), dtype=torch.int64)
                )

        if not torch.equal(has_text, chunk_counts > 0):
            raise RuntimeError(
                "Internal chunk accounting did not match non-empty texts"
            )

        nonempty_sums = sums[has_text]
        normalized = F.normalize(nonempty_sums, p=2, dim=1)
        if not torch.isfinite(normalized).all():
            raise RuntimeError("The model produced a non-finite embedding")

        results: list[list[float] | None] = []
        normalized_index = 0
        for is_nonempty in has_text.tolist():
            if not is_nonempty:
                results.append(None)
                continue
            results.append(normalized[normalized_index].tolist())
            normalized_index += 1
        return results


def discover_input_files(input_dir: Path) -> list[Path]:
    """Return source CSVs in deterministic filename order."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    paths = sorted(input_dir.glob("*.csv"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")
    return paths


def iter_metadata_rows(input_dir: Path) -> Iterator[MetadataRow]:
    """Stream validated metadata records from every input CSV."""

    for path in discover_input_files(input_dir):
        with path.open("r", encoding="utf-8-sig", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
                raise ValueError(
                    f"Unexpected columns in {path}: {reader.fieldnames!r}; "
                    f"expected {list(SOURCE_COLUMNS)!r}"
                )
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(
                        f"Malformed CSV row in {path} near line {line_number}"
                    )
                yield MetadataRow(
                    collection=path.stem,
                    title=row["title"] or "",
                    description=row["description"] or "",
                    transcript=row["transcript"] or "",
                )


def batched(rows: Iterable[MetadataRow], size: int) -> Iterator[list[MetadataRow]]:
    """Yield fixed-size row batches, including a final partial batch."""

    if size < 1:
        raise ValueError("row_batch_size must be at least 1")
    batch: list[MetadataRow] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def serialize_embedding(embedding: list[float] | None, expected_size: int) -> str:
    """Serialize one finite vector as a compact JSON array."""

    if embedding is None:
        return ""
    if len(embedding) != expected_size:
        raise ValueError(
            f"Expected an embedding of length {expected_size}, got {len(embedding)}"
        )
    if not all(math.isfinite(value) for value in embedding):
        raise ValueError("Cannot serialize a non-finite embedding")
    return json.dumps(embedding, separators=(",", ":"), allow_nan=False)


class ProgressBar:
    """A dependency-free terminal progress bar for embedding source rows."""

    def __init__(self, total: int, initial: int = 0) -> None:
        self.total = total
        self.completed = initial
        self.started_at = time.monotonic()
        self.closed = False
        self._render()

    def update(self, count: int) -> None:
        self.completed += count
        self._render()

    def close(self) -> None:
        if self.closed:
            return
        self._render()
        print(file=sys.stderr, flush=True)
        self.closed = True

    def _render(self) -> None:
        width = 30
        fraction = self.completed / self.total if self.total else 1.0
        filled = min(width, int(width * fraction))
        elapsed = time.monotonic() - self.started_at
        rate = (self.completed / elapsed) if elapsed else 0.0
        remaining = self.total - self.completed
        eta = remaining / rate if rate else 0.0
        bar = f"{'#' * filled}{'-' * (width - filled)}"
        print(
            f"\rEmbeddings [{bar}] {fraction:6.1%} "
            f"({self.completed:,}/{self.total:,}) {rate:.1f} rows/s "
            f"ETA {int(eta // 60):02d}:{int(eta % 60):02d}",
            end="",
            file=sys.stderr,
            flush=True,
        )


def count_metadata_rows(input_dir: Path) -> int:
    """Count rows using the same validation and ordering as embedding generation."""

    return sum(1 for _ in iter_metadata_rows(input_dir))


def resume_existing_output(
    output_path: Path, source_rows: Iterator[MetadataRow]
) -> tuple[int, dict[str, int]]:
    """Validate a partial output and return its completed row counts.

    Comparing every saved source field prevents silently resuming against changed,
    reordered, or different input files.
    """

    completed_rows = 0
    collection_counts: dict[str, int] = {}
    with output_path.open("r", encoding="utf-8", newline="") as partial_file:
        reader = csv.DictReader(partial_file)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError(
                f"Cannot resume {output_path}: unexpected columns "
                f"{reader.fieldnames!r}; expected {list(OUTPUT_COLUMNS)!r}"
            )
        for line_number, saved_row in enumerate(reader, start=2):
            if None in saved_row or any(column not in saved_row for column in OUTPUT_COLUMNS):
                raise ValueError(
                    f"Cannot resume {output_path}: malformed row near line "
                    f"{line_number}"
                )
            try:
                source_row = next(source_rows)
            except StopIteration as error:
                raise ValueError(
                    f"Cannot resume {output_path}: it has more rows than the input"
                ) from error
            saved_source = tuple(saved_row[column] for column in SOURCE_COLUMNS)
            source_values = (
                source_row.title,
                source_row.description,
                source_row.transcript,
            )
            if saved_row["collection"] != source_row.collection or saved_source != source_values:
                raise ValueError(
                    f"Cannot resume {output_path}: saved row {completed_rows + 1} "
                    "does not match the current input"
                )
            completed_rows += 1
            collection_counts[source_row.collection] = (
                collection_counts.get(source_row.collection, 0) + 1
            )
    return completed_rows, collection_counts


def generate_embeddings_csv(
    *,
    input_dir: Path,
    output_path: Path,
    embedder: TextEmbedder,
    row_batch_size: int,
) -> tuple[int, dict[str, int]]:
    """Create or resume the output CSV and return its row counts."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if not output_path.exists() and legacy_temporary_path.exists():
        os.replace(legacy_temporary_path, output_path)
        print(
            f"Moved legacy checkpoint from {legacy_temporary_path} to {output_path}",
            file=sys.stderr,
            flush=True,
        )

    expected_rows = count_metadata_rows(input_dir)
    source_rows = iter_metadata_rows(input_dir)

    if output_path.exists():
        total_rows, collection_counts = resume_existing_output(
            output_path, source_rows
        )
        print(
            f"Resuming from {total_rows:,} saved rows in {output_path}",
            file=sys.stderr,
            flush=True,
        )
        output_mode = "a"
    else:
        total_rows = 0
        collection_counts = {}
        output_mode = "w"

    progress = ProgressBar(expected_rows, initial=total_rows)

    try:
        with output_path.open(output_mode, encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
            if output_mode == "w":
                writer.writeheader()

            for row_batch in batched(source_rows, row_batch_size):
                texts = [
                    text
                    for row in row_batch
                    for text in (row.title, row.description, row.transcript)
                ]
                embeddings = embedder.embed_texts(texts)
                if len(embeddings) != len(texts):
                    raise RuntimeError(
                        "The embedder returned a different number of results than inputs"
                    )

                for index, row in enumerate(row_batch):
                    embedding_offset = index * len(SOURCE_COLUMNS)
                    title_embedding, description_embedding, transcript_embedding = (
                        embeddings[embedding_offset : embedding_offset + 3]
                    )
                    writer.writerow(
                        {
                            "collection": row.collection,
                            "title": row.title,
                            "description": row.description,
                            "transcript": row.transcript,
                            "title_embedding": serialize_embedding(
                                title_embedding, embedder.embedding_size
                            ),
                            "description_embedding": serialize_embedding(
                                description_embedding, embedder.embedding_size
                            ),
                            "transcript_embedding": serialize_embedding(
                                transcript_embedding, embedder.embedding_size
                            ),
                        }
                    )
                    total_rows += 1
                    collection_counts[row.collection] = (
                        collection_counts.get(row.collection, 0) + 1
                    )

                output_file.flush()
                os.fsync(output_file.fileno())
                progress.update(len(row_batch))

    except BaseException:
        progress.close()
        print(
            f"Embedding stopped. Progress through {total_rows:,} rows is saved in "
            f"{output_path}; run this command again to resume.",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        progress.close()

    return total_rows, collection_counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create intfloat/e5-base-v2 embeddings for metadata CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"directory containing per-collection CSVs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"destination CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="number of token chunks per model inference batch (default: 32)",
    )
    parser.add_argument(
        "--row-batch-size",
        type=int,
        default=256,
        help="number of source rows prepared before each write (default: 256)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device such as auto, cpu, cuda, cuda:0, or mps (default: auto)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.row_batch_size < 1:
        raise ValueError("--row-batch-size must be at least 1")

    device = resolve_device(args.device)
    print(f"Loading {MODEL_NAME} on {device}", flush=True)
    embedder = E5Embedder(device=device, batch_size=args.batch_size)
    total_rows, collection_counts = generate_embeddings_csv(
        input_dir=args.input_dir.resolve(),
        output_path=args.output.resolve(),
        embedder=embedder,
        row_batch_size=args.row_batch_size,
    )

    print(f"Wrote {total_rows:,} rows to {args.output.resolve()}")
    for collection, count in collection_counts.items():
        print(f"  {collection}: {count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
