# Automated Harm Analysis

Run Granite Guardian on a CSV file of question/answer pairs.

## Setup

1. Install the Granite Guardian model in **gguf** format.
2. Copy `.env.example` to `.env` and fill in the required values (for example, the model path).
3. Make sure you have `uv` installed.

## Run

```bash
uv run main.py \
  --in input.csv \
  --out output.csv
```

## Options

* `--q` – Question column name (default: `question`)
* `--a` – Answer column name (default: `answer`)
* `--in` – Input CSV file (required)
* `--out` – Output CSV file (required)
* `--keep` – Columns to keep from the input csv

### Example

```bash
uv run main.py \
  --q prompt \
  --a response \
  --in data.csv \
  --out results.csv \
  --keep source id
```

The output CSV will contain the selected input columns plus the harm analysis results.
