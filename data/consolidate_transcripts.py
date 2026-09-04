#!/usr/bin/env python3
"""Create normalized transcript CSVs from ``raw/`` into ``in/``.

The cleaned all-metadata outputs include title and description parsed from
their source fields. The filename metadata map pairs text-file collections
with their original metadata exports. Keep metadata-specific additions in
``add_metadata`` so source parsing remains unchanged when that work is added.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


def configure_csv_field_limit() -> None:
    """Allow the CSV reader to handle the long transcript fields in this dataset."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


configure_csv_field_limit()


@dataclass(frozen=True)
class CollectionSource:
    """Location and format of one collection's transcript records."""

    name: str
    kind: str
    relative_path: Path


COLLECTIONS: tuple[CollectionSource, ...] = (
    CollectionSource(
        "briggs-v-elliot", "text_files", Path("briggs-v-elliot/briggs-v-elliot-txts")
    ),
    CollectionSource("isleevy", "text_files", Path("isleevy/isleevy-txts")),
    # CollectionSource("mizell", "text_files", Path("mizell/mizell-txts")),
    CollectionSource("jadelaine", "text_files", Path("jadelaine/jadelaine-txts")),
    CollectionSource("jtmccain", "text_files", Path("jtmccain/jtmccain-txts")),
    CollectionSource(
        "eaadams",
        "tagged_metadata_csv",
        Path("eaadams/training/cleaned/D_eaa_h2_all_metadata.csv"),
    ),
    CollectionSource(
        "mccray",
        "tagged_metadata_csv",
        Path("mccray/training/cleaned/D_mcc_h2_all_metadata.csv"),
    ),
    CollectionSource(
        "scchr",
        "tagged_metadata_csv",
        Path("scchr/training/cleaned/D_mcc_h2_all_metadata.csv"),
    ),
    CollectionSource(
        "simkins",
        "tagged_metadata_csv",
        Path("simkins/training/cleaned/D_mcc_h2_all_metadata.csv"),
    ),
)

# Add text-file collections here to join filename IDs to their metadata exports.
FILENAME_METADATA_SOURCES: dict[str, Path] = {
    "briggs-v-elliot": Path("briggs-v-elliot/original_data/briggs_v_elliot-220313.csv"),
    "isleevy": Path("isleevy/original_data/isleevy-220329.csv"),
    "mizell": Path("mizell/original_data/mizell-237740.csv"),
    "jadelaine": Path("jadelaine/original_data/jadelaine-237739.csv"),
    "jtmccain": Path("jtmccain/original_data/jtmccain-237742.csv"),
}

RAW_DATA_DIRECTORY = Path("raw")

TEXT_FILENAME_PATTERN = re.compile(r"(?P<id>\d+)-transcript\.txt")
TRANSCRIPT_OUTPUT_FIELDS = ("transcript",)
METADATA_OUTPUT_FIELDS = ("title", "description", "transcript")
FILENAME_METADATA_OUTPUT_FIELDS = ("title", "description", "transcript")
COMBINED_OUTPUT_FIELDS = ("text",)
METADATA_SECTION_PATTERN = re.compile(
    r"^\[(Title|Description|Transcript|Date)\]:\s?", re.MULTILINE
)


class CollectionError(ValueError):
    """A collection cannot be normalized safely."""


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise CollectionError(f"Missing {description}: {path}")


def read_filename_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Load the SSID-keyed fields needed for filename-based metadata joins."""
    require_path(metadata_path, "filename metadata CSV")
    if not metadata_path.is_file():
        raise CollectionError(f"Filename metadata path is not a file: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        required_fields = {"SSID", "Title", "Date", "Description"}
        available_fields = set(reader.fieldnames or ())
        missing_fields = required_fields - available_fields
        if missing_fields:
            raise CollectionError(
                f"Missing required columns in {metadata_path}: {', '.join(sorted(missing_fields))}"
            )

        metadata_by_id: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            record_id = row["SSID"]
            if not record_id:
                raise CollectionError(
                    f"Missing SSID in {metadata_path} row {row_number}"
                )
            if record_id in metadata_by_id:
                raise CollectionError(f"Duplicate SSID in {metadata_path}: {record_id}")
            metadata_by_id[record_id] = {
                "title": row["Title"] or "",
                "date": row["Date"] or "",
                "description": row["Description"] or "",
            }
    return metadata_by_id


def read_text_files(
    collection: CollectionSource, source_dir: Path, metadata_path: Path
) -> list[dict[str, str]]:
    require_path(source_dir, f"transcript directory for {collection.name}")
    if not source_dir.is_dir():
        raise CollectionError(
            f"Transcript path for {collection.name} is not a directory: {source_dir}"
        )
    metadata_by_id = read_filename_metadata(metadata_path)

    records: list[dict[str, str]] = []
    for transcript_path in source_dir.glob("*-transcript.txt"):
        match = TEXT_FILENAME_PATTERN.fullmatch(transcript_path.name)
        if match is None:
            raise CollectionError(
                f"Malformed transcript filename for {collection.name}: {transcript_path.name}"
            )
        record_id = match.group("id")
        metadata = metadata_by_id.get(record_id)
        if metadata is None:
            raise CollectionError(
                f"No metadata match for {collection.name} transcript ID {record_id} in {metadata_path}"
            )
        records.append(
            {
                "id": record_id,
                **metadata,
                "transcript": transcript_path.read_text(encoding="utf-8"),
            }
        )

    if not records:
        raise CollectionError(
            f"No *-transcript.txt files found for {collection.name}: {source_dir}"
        )
    return records


def read_transcript_csv(
    collection: CollectionSource, source_path: Path
) -> list[dict[str, str]]:
    require_path(source_path, f"transcript CSV for {collection.name}")
    if not source_path.is_file():
        raise CollectionError(
            f"Transcript path for {collection.name} is not a file: {source_path}"
        )

    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        required_fields = {"text"}
        available_fields = set(reader.fieldnames or ())
        missing_fields = required_fields - available_fields
        if missing_fields:
            raise CollectionError(
                f"Missing required columns in {source_path}: {', '.join(sorted(missing_fields))}"
            )

        records: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            transcript = row["text"]
            if transcript is None:
                raise CollectionError(
                    f"Missing transcript field in {source_path} row {row_number}"
                )
            records.append({"transcript": transcript})

    if not records:
        raise CollectionError(
            f"No transcript records found for {collection.name}: {source_path}"
        )
    return records


def unpack_metadata_sections(
    source_path: Path, row_number: int, text: str
) -> dict[str, str]:
    """Parse the bracketed title, description, and transcript sections in a metadata row."""
    matches = list(METADATA_SECTION_PATTERN.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        value_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(text)
        )
        value = text[match.end() : value_end].strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1].replace('""', '"')
        sections[match.group(1)] = value

    if "Title" not in sections:
        raise CollectionError(
            f"Missing [Title] section in {source_path} row {row_number}"
        )
    return sections


def read_tagged_metadata_csv(
    collection: CollectionSource, source_path: Path
) -> list[dict[str, str]]:
    require_path(source_path, f"metadata CSV for {collection.name}")
    if not source_path.is_file():
        raise CollectionError(
            f"Metadata path for {collection.name} is not a file: {source_path}"
        )

    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        required_fields = {"text"}
        available_fields = set(reader.fieldnames or ())
        missing_fields = required_fields - available_fields
        if missing_fields:
            raise CollectionError(
                f"Missing required columns in {source_path}: {', '.join(sorted(missing_fields))}"
            )

        records: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            text = row["text"]
            if text is None:
                raise CollectionError(
                    f"Missing metadata text in {source_path} row {row_number}"
                )
            sections = unpack_metadata_sections(source_path, row_number, text)
            records.append(
                {
                    "title": sections["Title"],
                    "description": sections.get("Description", ""),
                    # Preserve an empty transcript when a source row has no [Transcript] section.
                    "transcript": sections.get("Transcript", ""),
                }
            )

    if not records:
        raise CollectionError(
            f"No metadata records found for {collection.name}: {source_path}"
        )
    return records


def validate_and_sort(
    collection: CollectionSource, records: Iterable[dict[str, str]]
) -> list[dict[str, str]]:
    records = list(records)
    if "id" not in records[0]:
        return records

    sorted_records = sorted(records, key=lambda record: int(record["id"]))
    seen_ids: set[str] = set()
    for record in sorted_records:
        record_id = record["id"]
        if not record_id.isdecimal():
            raise CollectionError(f"Non-numeric ID in {collection.name}: {record_id!r}")
        if record_id in seen_ids:
            raise CollectionError(f"Duplicate ID in {collection.name}: {record_id}")
        seen_ids.add(record_id)
    return sorted_records


def no_metadata_records(records: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Keep no-metadata outputs limited to transcripts."""
    return [
        {field: record[field] for field in TRANSCRIPT_OUTPUT_FIELDS}
        for record in records
    ]


def add_metadata(
    collection: CollectionSource, records: Sequence[dict[str, str]]
) -> list[dict[str, str]]:
    """Return metadata-ready rows; extend this function when mappings are available."""
    if collection.name in FILENAME_METADATA_SOURCES:
        return [
            {
                "title": title_with_date(record["title"], record["date"]),
                "description": record["description"],
                "transcript": record["transcript"],
            }
            for record in records
        ]
    if collection.kind == "tagged_metadata_csv":
        return [
            {field: record[field] for field in METADATA_OUTPUT_FIELDS}
            for record in records
        ]
    return [
        {field: record[field] for field in TRANSCRIPT_OUTPUT_FIELDS}
        for record in records
    ]


def title_with_date(title: str, date: str) -> str:
    """Append a text-file record's date to its title, except for undated records."""
    normalized_date = date.strip()
    if normalized_date.casefold() == "undated":
        return title
    return f"{title} ({normalized_date})"


def write_csv(
    path: Path, records: Iterable[dict[str, str]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)


def read_csv_records(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    """Read CSV records after confirming their required fields are present."""
    with path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        available_fields = set(reader.fieldnames or ())
        missing_fields = required_fields - available_fields
        if missing_fields:
            raise CollectionError(
                f"Missing required columns in {path}: {', '.join(sorted(missing_fields))}"
            )
        return [dict(row) for row in reader]


def metadata_text(record: dict[str, str]) -> str:
    """Serialize one metadata record using the requested literal XML-style tags."""
    sections: list[str] = []
    if record["title"]:
        sections.append(f"<title>\n{record['title']}\n</title>")
    if record["description"]:
        sections.append(f"<description>\n{record['description']}\n</description>")
    sections.append(f"<content>\n{record['transcript']}\n</content>")
    return "\n".join(sections)


def combined_records(
    input_directory: Path,
    required_fields: set[str],
    formatter: Callable[[dict[str, str]], str],
) -> list[dict[str, str]]:
    """Load and serialize every per-collection CSV in one input directory."""
    require_path(input_directory, "combined input directory")
    if not input_directory.is_dir():
        raise CollectionError(
            f"Combined input path is not a directory: {input_directory}"
        )

    input_paths = sorted(input_directory.glob("*.csv"))
    if not input_paths:
        raise CollectionError(
            f"No CSV files found in combined input directory: {input_directory}"
        )

    records: list[dict[str, str]] = []
    for input_path in input_paths:
        records.extend(
            {"text": formatter(record)}
            for record in read_csv_records(input_path, required_fields)
        )
    return records


def write_combined_csvs(root: Path, seed: int) -> None:
    """Write independently shuffled metadata and transcript-only training CSVs."""
    combined_sources = (
        (
            root / "in/metadata",
            root / "in/combined_metadata.csv",
            set(METADATA_OUTPUT_FIELDS),
            metadata_text,
        ),
        (
            root / "in/no_metadata",
            root / "in/combined_no_metadata.csv",
            set(TRANSCRIPT_OUTPUT_FIELDS),
            lambda record: record["transcript"],
        ),
    )
    for input_directory, output_path, required_fields, formatter in combined_sources:
        records = combined_records(input_directory, required_fields, formatter)
        random.Random(seed).shuffle(records)
        write_csv(output_path, records, COMBINED_OUTPUT_FIELDS)
        print(f"{output_path.name}: {len(records)} documents")


def collect_records(root: Path, collection: CollectionSource) -> list[dict[str, str]]:
    raw_root = root / RAW_DATA_DIRECTORY
    source_path = raw_root / collection.relative_path
    if collection.kind == "text_files":
        try:
            metadata_path = raw_root / FILENAME_METADATA_SOURCES[collection.name]
        except KeyError as error:
            raise CollectionError(
                f"No filename metadata source configured for {collection.name}"
            ) from error
        records = read_text_files(collection, source_path, metadata_path)
    elif collection.kind == "transcript_csv":
        records = read_transcript_csv(collection, source_path)
    elif collection.kind == "tagged_metadata_csv":
        records = read_tagged_metadata_csv(collection, source_path)
    else:
        raise CollectionError(
            f"Unsupported source type for {collection.name}: {collection.kind}"
        )
    return validate_and_sort(collection, records)


def main(root: Path, seed: int) -> None:
    normalized = {
        collection.name: collect_records(root, collection) for collection in COLLECTIONS
    }
    for collection in COLLECTIONS:
        records = normalized[collection.name]
        write_csv(
            root / "in/no_metadata" / f"{collection.name}.csv",
            no_metadata_records(records),
            TRANSCRIPT_OUTPUT_FIELDS,
        )
        metadata_fields = (
            FILENAME_METADATA_OUTPUT_FIELDS
            if collection.name in FILENAME_METADATA_SOURCES
            else METADATA_OUTPUT_FIELDS
            if collection.kind == "tagged_metadata_csv"
            else TRANSCRIPT_OUTPUT_FIELDS
        )
        write_csv(
            root / "in/metadata" / f"{collection.name}.csv",
            add_metadata(collection, records),
            metadata_fields,
        )
        print(f"{collection.name}: {len(records)} records")
    write_combined_csvs(root, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="data directory containing raw/ inputs and in/ outputs (default: script directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to deterministically shuffle each combined CSV (default: 42)",
    )
    arguments = parser.parse_args()
    try:
        main(arguments.root.resolve(), arguments.seed)
    except (CollectionError, OSError, UnicodeError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
