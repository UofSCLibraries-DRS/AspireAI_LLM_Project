#!/usr/bin/env python3
"""Load ``data/embeddings.csv`` into the local lighthouse_rag PostgreSQL database.

The loader uses psql's ``\\copy`` command, so the CSV is read by the machine
running this script rather than by the PostgreSQL server.  This is important
for a local database and avoids loading a large embeddings file into Python.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "data" / "embeddings.csv"
VECTOR_COLUMNS = (
    "title_embedding",
    "description_embedding",
    "transcript_embedding",
)
REQUIRED_COLUMNS = (
    "collection",
    "title",
    "description",
    "transcript",
    *VECTOR_COLUMNS,
)
DATABASE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def csv_field_limit() -> None:
    """Raise the CSV parser limit enough to accommodate long transcripts."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def inspect_csv(path: Path) -> int:
    """Read the header and one vector to determine the pgvector dimension.

    PostgreSQL validates the remaining rows while ``\\copy`` streams them. This
    avoids a second, very expensive full pass over a multi-gigabyte CSV.
    """
    csv_field_limit()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV is missing its header row")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            for column in VECTOR_COLUMNS:
                raw_vector = row[column].strip()
                if not raw_vector:
                    continue
                try:
                    vector = json.loads(raw_vector)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {column!r} on CSV line {line_number}"
                    ) from exc
                if not isinstance(vector, list) or not vector or not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in vector
                ):
                    raise ValueError(
                        f"{column!r} on CSV line {line_number} is not a numeric vector"
                    )
                return len(vector)

    raise ValueError("CSV contains no embeddings")


def run_psql(database: str, sql: str, args: argparse.Namespace) -> None:
    command = ["psql", "--no-psqlrc", "--set", "ON_ERROR_STOP=1", "--dbname", database]
    if args.host:
        command.extend(["--host", args.host])
    if args.port:
        command.extend(["--port", str(args.port)])
    if args.user:
        command.extend(["--username", args.user])
    subprocess.run(command, input=sql, text=True, check=True)


def psql_path(path: Path) -> str:
    """Quote a local path for psql's \\copy meta-command."""
    return str(path).replace("'", "''")


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a pgvector RAG database from embeddings.csv.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to embeddings.csv")
    parser.add_argument("--database", default="lighthouse_rag", help="PostgreSQL database name")
    parser.add_argument("--host", help="PostgreSQL host (defaults to local socket)")
    parser.add_argument("--port", type=int, help="PostgreSQL port")
    parser.add_argument("--user", help="PostgreSQL role (defaults to libpq's normal behavior)")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete existing rows in documents before loading (destructive)",
    )
    args = parser.parse_args()

    if not DATABASE_NAME.fullmatch(args.database):
        parser.error("--database must be a simple PostgreSQL identifier")
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        parser.error(f"CSV file does not exist: {csv_path}")

    print(f"Inspecting {csv_path}…", flush=True)
    try:
        dimensions = inspect_csv(csv_path)
    except (OSError, ValueError) as exc:
        print(f"Cannot import CSV: {exc}", file=sys.stderr)
        return 1
    print(f"Detected {dimensions}-dimensional vectors; starting streaming import.", flush=True)

    truncate = "TRUNCATE documents;" if args.replace else ""
    sql = f"""
BEGIN;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    collection TEXT NOT NULL,
    title TEXT,
    description TEXT,
    transcript TEXT,
    title_embedding VECTOR({dimensions}),
    description_embedding VECTOR({dimensions}),
    transcript_embedding VECTOR({dimensions})
);
{truncate}
\\copy documents (collection, title, description, transcript, title_embedding, description_embedding, transcript_embedding) FROM '{psql_path(csv_path)}' WITH (FORMAT csv, HEADER true, NULL '')
COMMIT;

SELECT count(*) AS documents_loaded FROM documents;
"""
    try:
        run_psql(args.database, sql, args)
    except FileNotFoundError:
        print("psql was not found. Install the PostgreSQL client tools and try again.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(
            "Import failed. Ensure the lighthouse_rag database exists, PostgreSQL is running, "
            "and the pgvector extension is installed.",
            file=sys.stderr,
        )
        return exc.returncode or 1

    print("Import complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
