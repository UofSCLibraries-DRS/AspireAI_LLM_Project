"""Report transcript-length statistics from an embeddings CSV.

The CSV is read one row at a time, so this is suitable for the multi-gigabyte
``data/embeddings.csv`` file. Character counts are Python Unicode character
counts (``len(transcript)``), not UTF-8 byte counts.
"""

import argparse
import csv
import heapq
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "data" / "embeddings.csv"
IDENTIFIER_COLUMNS = ("collection", "title")


def csv_field_limit() -> None:
    """Raise the parser's field limit enough to accommodate long transcripts."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report the mean and largest transcript character counts."
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, help="Path to embeddings.csv"
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Number of largest rows to report"
    )
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        parser.error(f"CSV file does not exist: {csv_path}")

    csv_field_limit()
    total_characters = 0
    row_count = 0
    largest: list[tuple[int, int, tuple[str, str]]] = []

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV is missing its header row")
            missing = {"transcript", *IDENTIFIER_COLUMNS} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {', '.join(sorted(missing))}"
                )

            for csv_row_number, row in enumerate(reader, start=2):
                transcript = row["transcript"] or ""
                character_count = len(transcript)
                total_characters += character_count
                row_count += 1
                identifiers = (
                    str(row["collection"] or ""),
                    str(row["title"] or ""),
                )
                entry = (
                    character_count,
                    csv_row_number,
                    identifiers,
                )
                if len(largest) < args.top:
                    heapq.heappush(largest, entry)
                elif entry[:2] > largest[0][:2]:
                    heapq.heapreplace(largest, entry)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"Cannot analyze CSV: {exc}", file=sys.stderr)
        return 1

    mean = total_characters / row_count if row_count else 0
    print(f"CSV: {csv_path}")
    print(f"Documents: {row_count:,}")
    print(f"Mean transcript characters: {mean:,.2f}")
    print()
    print(f"Top {min(args.top, row_count)} largest transcripts:")
    writer = csv.writer(sys.stdout)
    writer.writerow(("rank", "csv_row", "characters", *IDENTIFIER_COLUMNS))
    for rank, (character_count, csv_row_number, identifiers) in enumerate(
        sorted(largest, reverse=True), start=1
    ):
        writer.writerow((rank, csv_row_number, character_count, *identifiers))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
