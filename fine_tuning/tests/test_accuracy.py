import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from fine_tuning.evaluation.accuracy import NUBIA_METRIC_NAME, gaico_accuracy


class FakeExperiment:
    def __init__(self, llm_responses, reference_answer):
        self.responses = llm_responses
        self.reference_answer = reference_answer

    def compare(self, plot=False):
        assert not plot
        return pd.DataFrame(
            {
                "model_name": list(self.responses),
                "metric_name": ["ExistingMetric"] * len(self.responses),
                "score": [0.25] * len(self.responses),
            }
        )


class FakeNubia:
    def __init__(self):
        self.calls = []

    def score(self, ref, hyp):
        self.calls.append((ref, hyp))
        return {"first": 0.2, "second": 0.6}[hyp]


class AccuracyTest(unittest.TestCase):
    @patch("fine_tuning.evaluation.accuracy.Experiment", FakeExperiment)
    def test_nubia_is_scored_and_averaged_with_existing_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model"
            result = SimpleNamespace(
                model=str(model_path),
                prompt_template="prompts/base.yaml",
                responses=["first", "second"],
                ground_truth_short="short reference",
                ground_truth_ideal="ideal reference",
                ground_truth_short_agg="short aggregate reference",
                ground_truth_ideal_agg="ideal aggregate reference",
            )
            nubia = FakeNubia()

            gaico_accuracy([result], nubia_scorer=nubia)

            expected_references = {
                "short": "short reference",
                "ideal": "ideal reference",
                "short_agg": "short aggregate reference",
                "ideal_agg": "ideal aggregate reference",
            }
            for variant, reference in expected_references.items():
                summary_path = (
                    model_path
                    / "results"
                    / "base"
                    / f"metrics_summary_{variant}.csv"
                )
                with summary_path.open(newline="", encoding="utf-8") as handle:
                    metrics = {
                        row["metric"]: float(row["average"])
                        for row in csv.DictReader(handle)
                    }

                self.assertEqual(metrics["ExistingMetric"], 0.25)
                self.assertAlmostEqual(metrics[NUBIA_METRIC_NAME], 0.4)
                self.assertIn((reference, "first"), nubia.calls)
                self.assertIn((reference, "second"), nubia.calls)

            self.assertEqual(len(nubia.calls), 8)


if __name__ == "__main__":
    unittest.main()
