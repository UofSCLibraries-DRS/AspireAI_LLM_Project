# RAG evaluation pipeline

Run commands from this directory:

```sh
cd evaluation/rag_test
```

For experiments that use gated Hugging Face models, create the local environment
file and set `HF_TOKEN` in it:

```sh
cp .env.example .env
```

The Hugging Face account associated with that token must also have approved access
to every gated repository used by the experiment, including the configured
Meta-Llama repository. The pipeline checks for the token before changing an existing
results file.

Run an experiment JSON through response generation, GAICO evaluation, and figure
generation:

```sh
uv run python main.py configs/experiments/<experiment>.json
```

For example:

```sh
uv run python main.py configs/experiments/m14_naive_rag_greedy.json
```

Regenerate metrics and figures from an existing `results.csv` without running
inference:

```sh
uv run python main.py configs/experiments/m14_naive_rag_greedy.json --skip-inference
```

GAICO output includes BERTScore precision, recall, and F1, plus semantic cosine
similarity from Sentence Transformers using `all-MiniLM-L6-v2`.

Re-running the command preserves successful result rows and retries rows whose `error`
field is nonempty. Use `--k N` to request `N` responses for each question and chatbot
combination. Oversized-context rows remain errors and are retried on later runs.

The pipeline defaults vLLM workers to the `spawn` multiprocessing method. This is
required because experiments construct sequential vLLM engines after the E5/PyTorch
retriever has run; forking a new CUDA engine from that process can fail during CUDA
initialization.
## Retriever backends

The configured experiments run five retrieval variants for each chatbot:


- `NaiveRetriever` embeds the original question and performs pgvector search.
- `KeywordRetriever` asks the chatbot for a small keyword list and performs BM25 search.
- `HybridRetriever` independently ranks vector and keyword candidates, then combines
  their ranks with reciprocal-rank fusion (RRF).
- `HyDEAnswerRetriever` generates a hypothetical answer and embeds it for vector
  search.
- `HyDEDocumentRetriever` generates a hypothetical source passage and embeds it for
  vector search.

The keyword and hybrid retrievers require the `pg_textsearch` extension and the
`documents_search_text_bm25` index. After loading embeddings, create or update the
indexes from the repository root:

```sh
cd demo/backend
psql -d lighthouse_rag -v ON_ERROR_STOP=1 -f db/schema.sql
```

Keyword and HyDE generation use the same chatbot being evaluated. Their token limits,
prompt templates, BM25 index name, RRF constant, and candidate multiplier can be
overridden in each retriever's experiment config. The M14 models are instruction
models, so the default retriever prompts are direct user instructions with explicit
output constraints and delimited question text. `VLLMChatbot` applies the model's
native chat template; do not add model-specific chat tokens to retriever prompts.

HyDE defaults to E5's `passage: `
prefix because it embeds generated passage-like text; naive and hybrid query search
use the `query: ` prefix.

## Rank existing model scores

Each GAICO CSV contains the raw value for every metric. BERTScore F1 is
retained; BERTScore precision and recall are omitted.

Rank models from the GAICO CSVs without rerunning inference or metrics:

```sh
uv run python utils/rank_models.py results/M14_greedy/gaico \
  --top 10 \
  --output results/M14_greedy/model_rankings.csv
```

By default, the script uses every available metric. It averages repeated responses per
scenario and keeps only scenarios scored for every model. Rankings are calculated
independently for each input score CSV (`answer_ideal`, `answer_ideal_agg`,
`answer_short`, and `answer_short_agg`), with rank 1 restarting for every file. Within
each score subset, every available metric receives equal weight. The output CSV uses
the input filename stem as `subset` and includes the raw overall and per-metric means.
Use repeated `--metric NAME` arguments to rank on selected metrics only. All metrics
currently emitted by this project are higher-is-better; `--lower-is-better NAME` is
available for future distance or loss metrics.

Group those rankings into RAG-type by fine-tuning-status matrices:

```sh
uv run python utils/group_result_matrices.py
```

This writes `results/M14_greedy/rag_finetune_average.csv` and
`results/M14_greedy/rag_finetune_max.csv`. Each score subset is retained, RAG types
form the rows, and `fine_tuned` / `not_fine_tuned` form the value columns.
The default cell value is `ranking_score`; the average matrix averages model
scores within each cell, while the maximum matrix reports both the highest
model score and the winning model ID.

The command also writes color-coded average and maximum heatmaps beneath
`results/M14_greedy/figures/rag_vs_fine-tuning/`
`{ideal,ideal_agg,short,short_agg}`. To plot another numeric ranking column
instead, use:

```sh
uv run python utils/group_result_matrices.py --value-column mean_score
```

Create per-subset horizontal bar charts for every metric:

```sh
uv run python utils/metric_bar_charts.py
```

Charts are written beneath
`results/M14_greedy/metric_bar_charts/{ideal,ideal_agg,short,short_agg}`.
Each subset also receives `configuration_mean.png`, which first averages all
available metric means within each model/configuration and then plots those
configuration-level values.
