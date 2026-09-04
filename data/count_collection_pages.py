#!/usr/bin/env python3
"""Count unique page bodies in the selected raw transcript collections.

Each ``=== Page N ===`` line starts a page. Pages are deduplicated by their
text between markers, after leading and trailing whitespace is removed.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CollectionSource:
    """A text-file collection relative to the raw-data directory."""

    name: str
    kind: str
    relative_path: Path


COLLECTIONS: tuple[CollectionSource, ...] = (
    CollectionSource(
        "briggs-v-elliot", "text_files", Path("briggs-v-elliot/briggs-v-elliot-txts")
    ),
    CollectionSource("isleevy", "text_files", Path("isleevy/isleevy-txts")),
    CollectionSource("mizell", "text_files", Path("mizell/mizell-txts")),
    CollectionSource("jadelaine", "text_files", Path("jadelaine/jadelaine-txts")),
    CollectionSource("jtmccain", "text_files", Path("jtmccain/jtmccain-txts")),
)

PAGE_MARKER_PATTERN = re.compile(r"^=== Page \d+ ===[ \t]*$", re.MULTILINE)


def count_pages(collection_dir: Path) -> tuple[int, int, int]:
    """Return the number of files, raw pages, and unique pages."""
    transcript_paths = sorted(collection_dir.glob("*-transcript.txt"))
    raw_page_count = 0
    unique_page_texts: set[str] = set()

    for transcript_path in transcript_paths:
        transcript = transcript_path.read_text(encoding="utf-8")
        page_texts = PAGE_MARKER_PATTERN.split(transcript)[1:]
        raw_page_count += len(page_texts)
        unique_page_texts.update(page_text.strip() for page_text in page_texts)

    return len(transcript_paths), raw_page_count, len(unique_page_texts)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Count distinct page bodies in the selected transcript collections."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=script_dir / "raw",
        help="Raw-data directory. Defaults to the raw directory beside this script.",
    )
    args = parser.parse_args()

    results: list[tuple[str, int, int, int]] = []
    for collection in COLLECTIONS:
        collection_dir = args.raw_dir / collection.relative_path
        if not collection_dir.is_dir():
            parser.error(f"Missing transcript directory for {collection.name}: {collection_dir}")
        files, raw_pages, unique_pages = count_pages(collection_dir)
        results.append((collection.name, files, raw_pages, unique_pages))

    name_width = max(len("Collection"), *(len(name) for name, *_ in results))
    print(
        f"{'Collection':<{name_width}}  {'Files':>5}  {'Raw Count':>9}  "
        f"{'Unique Count':>12}  {'Duplicate Pages':>15}"
    )
    print(
        f"{'-' * name_width}  {'-' * 5}  {'-' * 9}  {'-' * 12}  {'-' * 15}"
    )
    for name, files, raw_pages, unique_pages in results:
        print(
            f"{name:<{name_width}}  {files:>5}  {raw_pages:>9}  "
            f"{unique_pages:>12}  {raw_pages - unique_pages:>15}"
        )
    total_files = sum(row[1] for row in results)
    total_raw_pages = sum(row[2] for row in results)
    total_unique_pages = sum(row[3] for row in results)
    print(
        f"{'Total':<{name_width}}  {total_files:>5}  {total_raw_pages:>9}  "
        f"{total_unique_pages:>12}  {total_raw_pages - total_unique_pages:>15}"
    )


if __name__ == "__main__":
    main()
