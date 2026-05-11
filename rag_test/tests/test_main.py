import argparse
import unittest

from main import positive_int


class MainCliTests(unittest.TestCase):
    def test_positive_int_accepts_positive_values(self) -> None:
        self.assertEqual(positive_int("3"), 3)

    def test_positive_int_rejects_non_positive_values(self) -> None:
        for value in ("0", "-1", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    positive_int(value)


if __name__ == "__main__":
    unittest.main()
