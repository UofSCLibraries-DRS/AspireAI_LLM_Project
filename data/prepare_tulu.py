#!/usr/bin/env python3

import csv
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


# =============================================================================
# Hard-coded settings
# =============================================================================

INPUT_DIR = Path("./tulu-3-sft-mixture-english")

TOKENIZER_PATH = Path("/work/jaaydin/models/Llama-3.1-8B-Instruct")

OUTPUT_CSV = Path("./tulu_llama_5000chars_50000.csv")

MAX_CHARACTERS = 5_000
SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

# Increase this if the machine has enough CPU cores.
NUM_PROC = 1

# Number of examples processed in each tokenizer-template batch.
MAP_BATCH_SIZE = 1_000

LLAMA_BOS_TOKEN = "<|begin_of_text|>"
LLAMA_EOT_TOKEN = "<|eot_id|>"
NUL_CHARACTER = "\x00"


# =============================================================================
# Processing functions
# =============================================================================

_tokenizer = None


def remove_nul_characters(text):
    """Remove control characters that Python's CSV reader cannot accept."""
    return text.replace(NUL_CHARACTER, "")


def get_tokenizer():
    """Load and validate the Llama tokenizer once per process."""
    global _tokenizer

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            str(TOKENIZER_PATH),
            local_files_only=True,
            use_fast=True,
        )

        if not _tokenizer.chat_template:
            raise ValueError(
                f"The tokenizer at {TOKENIZER_PATH} does not contain "
                "a chat template."
            )

        if _tokenizer.bos_token != LLAMA_BOS_TOKEN:
            raise ValueError(
                f"Expected Llama BOS token {LLAMA_BOS_TOKEN!r}, found "
                f"{_tokenizer.bos_token!r}."
            )

        if LLAMA_EOT_TOKEN not in _tokenizer.get_vocab():
            raise ValueError(
                f"The tokenizer at {TOKENIZER_PATH} does not contain "
                f"{LLAMA_EOT_TOKEN}."
            )

    return _tokenizer


def normalize_messages(messages):
    """Convert a conversation to the role/content format used by the template."""
    normalized = []

    if messages is None:
        return normalized

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role is None or content is None:
            continue

        normalized.append(
            {
                "role": remove_nul_characters(str(role)),
                "content": remove_nul_characters(str(content)),
            }
        )

    return normalized


def ensure_terminal_eot(text):
    """Remove trailing whitespace and ensure a rendered chat ends with EOT."""
    normalized = text.rstrip()
    if not normalized.endswith(LLAMA_EOT_TOKEN):
        normalized += LLAMA_EOT_TOKEN
    return normalized


def apply_llama_template(batch):
    """Apply the Llama 3.1 Instruct chat template to a batch of conversations."""
    tokenizer = get_tokenizer()

    formatted_texts = []
    character_counts = []
    valid_rows = []

    for messages in batch["messages"]:
        normalized_messages = normalize_messages(messages)

        if not normalized_messages:
            formatted_texts.append("")
            character_counts.append(0)
            valid_rows.append(False)
            continue

        try:
            text = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            text = remove_nul_characters(text)
            text = ensure_terminal_eot(text)

            if not text.startswith(LLAMA_BOS_TOKEN):
                raise ValueError(
                    f"Rendered conversation does not start with {LLAMA_BOS_TOKEN}."
                )
        except Exception as exc:
            print(f"Skipping malformed conversation: {exc}")
            formatted_texts.append("")
            character_counts.append(0)
            valid_rows.append(False)
            continue

        formatted_texts.append(text)
        character_counts.append(len(text))
        valid_rows.append(True)

    return {
        "text": formatted_texts,
        "character_count": character_counts,
        "valid_row": valid_rows,
    }


def validate_output_csv():
    """Verify the exported CSV can be read and every row has Llama boundaries."""
    row_count = 0
    with OUTPUT_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        if reader.fieldnames != ["text"]:
            raise ValueError(
                f"Expected CSV columns ['text'], found {reader.fieldnames}."
            )

        for row_count, row in enumerate(reader, start=1):
            text = row["text"]
            if not text.startswith(LLAMA_BOS_TOKEN):
                raise ValueError(f"CSV row {row_count} is missing the Llama BOS token.")
            if not text.endswith(LLAMA_EOT_TOKEN):
                raise ValueError(f"CSV row {row_count} is missing the terminal EOT.")
            if len(text) > MAX_CHARACTERS:
                raise ValueError(
                    f"CSV row {row_count} has {len(text):,} characters; "
                    f"the limit is {MAX_CHARACTERS:,}."
                )

    if row_count != SAMPLE_SIZE:
        raise ValueError(f"Expected {SAMPLE_SIZE:,} CSV rows, found {row_count:,}.")


# =============================================================================
# Main program
# =============================================================================


def main():
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {INPUT_DIR.resolve()}"
        )

    if not TOKENIZER_PATH.is_dir():
        raise FileNotFoundError(
            f"Tokenizer directory does not exist: {TOKENIZER_PATH.resolve()}"
        )

    parquet_files = sorted(INPUT_DIR.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"No Parquet files were found under: {INPUT_DIR.resolve()}"
        )

    print(f"Input directory: {INPUT_DIR.resolve()}")
    print(f"Tokenizer:       {TOKENIZER_PATH.resolve()}")
    print(f"Output CSV:      {OUTPUT_CSV.resolve()}")
    print(f"Parquet files:   {len(parquet_files):,}")
    print()

    # Load the tokenizer now so configuration errors are caught before the
    # full dataset is processed.
    tokenizer = get_tokenizer()

    print("Loaded tokenizer.")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print()

    print("Loading dataset...")

    dataset = load_dataset(
        "parquet",
        data_files=[str(file) for file in parquet_files],
        split="train",
    )

    print(f"Loaded rows: {len(dataset):,}")

    if "messages" not in dataset.column_names:
        raise KeyError(
            "The dataset does not contain a 'messages' column. "
            f"Available columns: {dataset.column_names}"
        )

    original_columns = dataset.column_names

    print("Applying the Llama 3.1 Instruct chat template...")

    formatted = dataset.map(
        apply_llama_template,
        batched=True,
        batch_size=MAP_BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=original_columns,
        desc="Applying chat template",
    )

    print(f"Filtering formatted text to at most {MAX_CHARACTERS:,} characters...")

    filtered = formatted.filter(
        lambda row: (row["valid_row"] and 0 < row["character_count"] <= MAX_CHARACTERS),
        num_proc=NUM_PROC,
        desc="Filtering by formatted length",
    )

    eligible_count = len(filtered)

    print(f"Eligible rows: {eligible_count:,}")

    if eligible_count < SAMPLE_SIZE:
        raise ValueError(
            f"Only {eligible_count:,} rows passed the filter, but "
            f"{SAMPLE_SIZE:,} rows were requested."
        )

    print(f"Randomly sampling {SAMPLE_SIZE:,} rows with seed {RANDOM_SEED}...")

    sampled = filtered.shuffle(seed=RANDOM_SEED).select(range(SAMPLE_SIZE))

    # Retain only the requested CSV column.
    sampled = sampled.select_columns(["text"])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Writing CSV...")

    sampled.to_csv(str(OUTPUT_CSV), index=False)
    validate_output_csv()

    print()
    print("Done.")
    print(f"Rows written: {len(sampled):,}")
    print(f"CSV columns:  {sampled.column_names}")
    print(f"Output file:  {OUTPUT_CSV.resolve()}")
    print("Validated every row has Llama BOS and terminal EOT tokens.")


if __name__ == "__main__":
    main()
