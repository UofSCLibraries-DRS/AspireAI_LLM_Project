import csv
import tempfile
import unittest
from pathlib import Path

from fine_tuning.utils.training_data import (
    MAX_SEQUENCE_LENGTH,
    load_text_column,
)


class LoRAUnsupervisedTest(unittest.TestCase):
    def test_uses_2048_token_sequences(self):
        self.assertEqual(MAX_SEQUENCE_LENGTH, 2_048)

    def test_loads_text_fields_larger_than_the_default_csv_limit(self):
        long_text = "A" * 150_000 + "\nembedded newline, and comma"
        original_limit = csv.field_size_limit()

        try:
            csv.field_size_limit(131_072)
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = Path(temp_dir) / "training.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["text"])
                    writer.writeheader()
                    writer.writerow({"text": long_text})

                self.assertEqual(load_text_column(str(csv_path)), [long_text])
        finally:
            csv.field_size_limit(original_limit)


if __name__ == "__main__":
    unittest.main()
