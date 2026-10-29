"""Check a trained SafeChat model's actual replies against every CSV answer."""

import argparse
import asyncio
import csv
import json
from pathlib import Path

from prepare_scenarios import PROJECT, select_answer


async def evaluate(model, csv_path, report_path):
    from rasa.core.agent import Agent

    agent = Agent.load(str(model))
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    results = []
    for index, row in enumerate(rows):
        question = row["question"].strip()
        column, expected = select_answer(row)
        replies = await agent.handle_text(question, sender_id="scenario-check-{}".format(index))
        actual = "\n".join(reply.get("text", "") for reply in replies or [])
        result = {"question": question, "answer_column": column, "passed": actual == expected}
        if not result["passed"]:
            result.update(expected=expected, actual=actual)
        results.append(result)
        if (index + 1) % 20 == 0:
            print("Checked {}/{} questions".format(index + 1, len(rows)), flush=True)
    passed = sum(result["passed"] for result in results)
    report = {"model": str(model), "total": len(rows), "passed": passed,
              "failed": len(rows) - passed, "results": results}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Exact answer matches: {}/{}. Report: {}".format(passed, len(rows), report_path))
    return passed == len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=PROJECT / "scenarios_with_answers.csv")
    parser.add_argument("--report", type=Path, default=PROJECT / "results/scenarios_evaluation.json")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.model, args.csv, args.report)) else 1)
