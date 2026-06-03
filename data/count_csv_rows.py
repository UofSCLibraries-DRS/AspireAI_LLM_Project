#!/usr/bin/env python3
"""Print row counts for CSV files in a terminal table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def count_rows(path: Path, count_header: bool) -> int:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        row_count = sum(1 for _ in csv.reader(csv_file))

    if count_header or row_count == 0:
        return row_count
    return row_count - 1


def find_csv_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []

    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() == ".csv"
    )


def print_table(rows: list[tuple[str, int]]) -> None:
    file_width = max(len("CSV"), *(len(file_name) for file_name, _ in rows))
    count_width = max(len("Rows"), *(len(str(count)) for _, count in rows))

    separator = f"+-{'-' * file_width}-+-{'-' * count_width}-+"
    print(separator)
    print(f"| {'CSV'.ljust(file_width)} | {'Rows'.rjust(count_width)} |")
    print(separator)
    for file_name, count in rows:
        print(f"| {file_name.ljust(file_width)} | {str(count).rjust(count_width)} |")
    print(separator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count rows in CSV files and print the results in a table."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory to scan recursively, or a single CSV file. Defaults to the current directory.",
    )
    parser.add_argument(
        "--count-header",
        action="store_true",
        help="Include the header row in the count. By default, the first row is excluded.",
    )
    args = parser.parse_args()

    target = Path(args.path)
    csv_files = find_csv_files(target)
    display_root = target.parent if target.is_file() else target

    if not csv_files:
        print(f"No CSV files found in {target}.")
        return

    rows = [
        (str(path.relative_to(display_root)), count_rows(path, args.count_header))
        for path in csv_files
    ]
    print_table(rows)


if __name__ == "__main__":
    main()

