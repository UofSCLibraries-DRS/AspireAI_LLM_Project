import csv
import sys


MAX_SEQUENCE_LENGTH = 2_048


def configure_csv_field_limit() -> None:
    """Allow the CSV parser to read long transcript fields."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_text_column(data_path: str) -> list[str]:
    """Read the required text column after lifting Python's CSV field limit."""
    configure_csv_field_limit()

    with open(data_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames

        if columns is None or "text" not in columns:
            raise ValueError(
                f"CSV must contain a 'text' column. Columns found: {columns}"
            )

        return [str(row["text"]) for row in reader if row.get("text") is not None]
