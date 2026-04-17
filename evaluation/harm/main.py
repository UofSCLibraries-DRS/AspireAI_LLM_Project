import csv
import os
from dotenv import load_dotenv
from tqdm import tqdm

from src import HarmWrapper

load_dotenv()


def main(
    q_col: str,
    a_col: str,
    in_csv: str,
    out_csv: str,
    max_ctx=2048,
    keep_cols=[],
):
    model_path = os.getenv("GRANITE_GUARDIAN_PATH")

    if not model_path:
        raise ValueError("GRANITE_GUARDIAN_PATH not found in .env file")

    # Initialize the harm wrapper
    print(f"Loading model from {model_path}...")
    harm_checker = HarmWrapper(model_path=model_path, max_ctx=max_ctx)

    # Read input CSV
    print(f"Reading input from {in_csv}...")
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Process first row to determine harm columns
    harm_columns = []
    if rows:
        query = rows[0].get(q_col, "")
        answer = rows[0].get(a_col, "")
        harm_results = harm_checker.evaluate_harm(query=query, response=answer)
        harm_columns = list(harm_results.keys())

    keep_cols = [c for c in keep_cols if c not in ("question", "answer")]

    # Define new fieldnames: question, answer, and harm columns
    new_fieldnames = ["question", "answer"] + keep_cols + harm_columns

    # Process each row
    print(f"Processing {len(rows)} rows...")
    output_rows = []
    for row in tqdm(rows, desc="Evaluating harm"):
        query = row.get(q_col, "")
        answer = row.get(a_col, "")

        # Evaluate harm
        harm_results = harm_checker.evaluate_harm(query=query, response=answer)

        # Create new row with only question, answer, and harm results
        new_row = {
            "question": query,
            "answer": answer,
        }

        for col in keep_cols:
            new_row[col] = row.get(col, "")

        # Add harm results
        for criterion, is_harmful in harm_results.items():
            new_row[criterion] = "yes" if is_harmful else "no"

        output_rows.append(new_row)

    # Write output CSV
    print(f"Writing output to {out_csv}...")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Q&A pairs for harmful content using Granite Guardian"
    )
    parser.add_argument(
        "--q", type=str, default="question", help="Column name for questions"
    )
    parser.add_argument(
        "--a", type=str, default="answer", help="Column name for answers"
    )
    parser.add_argument(
        "--in", dest="in_csv", required=True, help="Input CSV file path"
    )
    parser.add_argument(
        "--out", dest="out_csv", required=True, help="Output CSV file path"
    )
    parser.add_argument(
        "--max-ctx", type=int, default=2048, help="Maximum context size"
    )
    parser.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help="Original CSV columns to keep in the output",
    )

    args = parser.parse_args()

    main(
        q_col=args.q,
        a_col=args.a,
        in_csv=args.in_csv,
        out_csv=args.out_csv,
        max_ctx=args.max_ctx,
        keep_cols=args.keep,
    )
