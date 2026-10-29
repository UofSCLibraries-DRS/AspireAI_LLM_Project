"""Update SafeChat's Rasa training data from scenarios_with_answers.csv."""

import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


PROJECT = Path(__file__).resolve().parent


def normalize(question):
    return re.sub(r"[^a-z0-9]+", " ", question.casefold().replace("'", "").replace("’", "")).strip()


def select_answer(row):
    for column in ("answer_ideal_agg", "answer_ideal"):
        answer = row.get(column, "").strip()
        if answer:
            return column, answer
    raise ValueError("Question has no ideal answer: {!r}".format(row["question"]))


def prepare(project=PROJECT, csv_path=None):
    project = Path(project)
    csv_path = Path(csv_path) if csv_path else project / "scenarios_with_answers.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"question", "answer_ideal", "answer_ideal_agg", "dataset"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain {}".format(sorted(required)))
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no scenarios")

    yaml = YAML()
    yaml.preserve_quotes = True
    domain = yaml.load(project / "domain.yml")
    nlu = yaml.load(project / "data/nlu.yml")
    rules = yaml.load(project / "data/rules.yml")
    examples = defaultdict(set)
    by_intent = {entry["intent"]: entry for entry in nlu["nlu"]}
    for entry in nlu["nlu"]:
        for line in entry["examples"].splitlines():
            if line.strip().startswith("- "):
                examples[normalize(line.strip()[2:])].add(entry["intent"])

    seen = set()
    assigned_intents = set()
    records = []
    for row in rows:
        question = row["question"].strip()
        key = normalize(question)
        if not key or key in seen:
            raise ValueError("Blank or duplicate question: {!r}".format(question))
        seen.add(key)
        column, answer = select_answer(row)
        candidates = examples[key]
        if len(candidates) > 1:
            raise ValueError("Question matches multiple intents: {!r}".format(question))
        intent = next(iter(candidates), None)
        if intent is None:
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
            dataset = re.sub(r"[^a-z0-9]+", "_", row["dataset"].casefold()).strip("_")
            intent = "scenario_{}_{}".format(dataset, digest)
        if intent in assigned_intents:
            raise ValueError("Different CSV questions map to the same intent: {}".format(intent))
        assigned_intents.add(intent)
        if intent not in by_intent:
            entry = {"intent": intent, "examples": LiteralScalarString("- " + question + "\n")}
            nlu["nlu"].append(entry)
            by_intent[intent] = entry
            domain["intents"].append(intent)
            rules["rules"].append({
                "rule": "Answer " + intent,
                "steps": [{"intent": intent}, {"action": "utter_" + intent}],
            })
        else:
            entry = by_intent[intent]
            existing = [line.strip()[2:] for line in entry["examples"].splitlines()
                        if line.strip().startswith("- ")]
            if question not in existing:
                entry["examples"] = LiteralScalarString(
                    entry["examples"].rstrip() + "\n- " + question + "\n"
                )

        domain["responses"]["utter_" + intent] = [{"text": answer}]
        summary = "utter_summary_" + intent
        if summary in domain["responses"]:
            # Existing two-step stories must give the selected answer on the first reply too.
            domain["responses"][summary] = [{"text": answer}]
        records.append({"question": question, "intent": intent, "column": column, "answer": answer})

    for path, document in (
        (project / "domain.yml", domain),
        (project / "data/nlu.yml", nlu),
        (project / "data/rules.yml", rules),
    ):
        with path.open("w", encoding="utf-8") as destination:
            yaml.dump(document, destination)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT / "scenarios_with_answers.csv")
    args = parser.parse_args()
    records = prepare(csv_path=args.csv)
    aggregate = sum(record["column"] == "answer_ideal_agg" for record in records)
    print("Prepared {} questions: {} ideal_agg, {} ideal.".format(
        len(records), aggregate, len(records) - aggregate
    ))
