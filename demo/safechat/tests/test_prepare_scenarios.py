import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from ruamel.yaml import YAML

from prepare_scenarios import PROJECT, prepare, select_answer


class ScenarioPreparationTests(unittest.TestCase):
    def test_prefers_aggregate_and_falls_back_to_ideal(self):
        self.assertEqual(select_answer({
            "answer_ideal_agg": " Aggregate ", "answer_ideal": "Ideal"
        }), ("answer_ideal_agg", "Aggregate"))
        self.assertEqual(select_answer({
            "answer_ideal_agg": "  ", "answer_ideal": "Ideal"
        }), ("answer_ideal", "Ideal"))
        with self.assertRaises(ValueError):
            select_answer({"question": "Unanswered", "answer_ideal": ""})

    def test_every_csv_question_has_its_selected_response_and_import_is_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "data").mkdir()
            paths = ["domain.yml", "data/nlu.yml", "data/rules.yml"]
            for name in paths + ["scenarios_with_answers.csv"]:
                shutil.copyfile(PROJECT / name, project / name)
            records = prepare(project)
            yaml = YAML(typ="safe")
            domain = yaml.load(project / "domain.yml")
            nlu = yaml.load(project / "data/nlu.yml")
            by_intent = {entry["intent"]: entry for entry in nlu["nlu"]}
            with (project / "scenarios_with_answers.csv").open(newline="", encoding="utf-8-sig") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(records), len(rows))
            for row, record in zip(rows, records):
                with self.subTest(question=row["question"]):
                    expected = row["answer_ideal_agg"].strip() or row["answer_ideal"].strip()
                    self.assertEqual(record["answer"], expected)
                    self.assertIn("- " + row["question"].strip(),
                                  by_intent[record["intent"]]["examples"].splitlines())
                    self.assertEqual(domain["responses"]["utter_" + record["intent"]],
                                     [{"text": expected}])
                    summary = "utter_summary_" + record["intent"]
                    if summary in domain["responses"]:
                        self.assertEqual(domain["responses"][summary], [{"text": expected}])
            subscriptions = [record for record in records if "subscription" in record["question"]]
            self.assertEqual(len({record["intent"] for record in subscriptions}), 2)
            first = {name: (project / name).read_bytes() for name in paths}
            self.assertEqual(prepare(project), records)
            self.assertEqual(first, {name: (project / name).read_bytes() for name in paths})


if __name__ == "__main__":
    unittest.main()
