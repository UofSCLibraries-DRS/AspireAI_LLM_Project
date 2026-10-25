## How to Run

### 1. Create virtual environment & install dependencies

```bash
uv sync
```

### 2. Start API server 
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The `RAG` model uses keyword-based BM25 retrieval from the local `lighthouse_rag_pages`
PostgreSQL database. It uses the fine-tuned `M14` Bedrock model to generate search terms,
then answer using the retrieved documents. Its optional environment settings
are documented in `.env.example`.

On EC2, set the backend's effective environment (or `demo/backend/.env`) to
`RAG_DATABASE_URL=dbname=lighthouse_rag_pages port=5433 user=jaaydin`, then
restart the backend. PostgreSQL 17 runs on port 5433 there; port 5432 still
serves PostgreSQL 16. Local development uses the pages database on port 5432.
Retrieval queries explicitly read `public.documents`. The EC2 role `jaaydin`
needs `SELECT` on that table because the restore created it as `postgres`.

### Fine-tuned Bedrock model (M14)

`M14` uses `M14-meta-e2-no-MLP-r16`, imported in `us-east-1` with ARN
`arn:aws:bedrock:us-east-1:777384741382:imported-model/jyve0zcux4qi`.
Select it with `"model": "M14"` in `/generate` or the frontend's
`Llama-3.1-8B-Instruct, Fine-tuned (M14)` option. M9 is no longer an API option.

The model config is `configs/chatbots/m14_bedrock.yaml`. Its prompt template,
`configs/prompt_templates/m14.yaml`, uses Llama 3.1 Instruct user/assistant
role headers and end-of-turn tokens without adding a system message.
We format prompts explicitly through `InvokeModel`, including when Bedrock
reports `instructSupported: false`. Stop sequences trim returned text locally.

The original `configs/chatbots/m9_bedrock.yaml` and
`configs/prompt_templates/m9.yaml` are preserved. To restore M9, change the
fine-tuned chatbot's ID, registry key, and request model literal in
`src/api/main.py` from `M14` to `M9`, use
`configs/chatbots/m9_bedrock.yaml` as its `config_path`, and update the
frontend option's name and `apiValue` accordingly. Restart the backend.

Run offline Bedrock regression tests with:

```bash
uv run python -m unittest discover -s tests -p test_bedrock.py
```

### Warm M14 when the demo opens

The deployed page's inline script in `../frontend/index.html` sends
`POST /api/warmup` when it opens. The backend starts a background, one-token
M14 inference and immediately returns `202` while warming or `200` when ready.
`GET /api/warmup` reads status without inference. Both endpoints return
`{"model":"M14","status":"cold|warming|ready|failed","message":null}`
(a safe message is provided on failure) with `Cache-Control: no-store`.
An unavailable coordinator returns `503`; POST also returns `503` during a
failure's 30-second cooldown.

M14 and RAG share readiness and a host-wide Linux file lock. Both Gunicorn
workers must use the same local state directory. A worker crash automatically
releases the lock, allowing another visitor to restart warm-up. State is keyed
by model ARN and region; a successful inference counts as ready for four
minutes. Each successful M14 call, including RAG keyword and answer calls,
refreshes this timestamp. Merely polling status does not refresh it.

The browser checks readiness before M14/RAG questions and retains one pending
question in memory while polling every two seconds. It sends the question once
ready, guards both click and Enter submissions, and restores the question after
failure for retry. A page refresh discards the pending question. Other models
bypass the gate. Direct API callers receive `503` with
`detail.code="model_warming"` and `Retry-After: 2` until ready; retry their
original `/generate` request after readiness is confirmed. Successful generation
responses retain their existing shape.

Warm-up uses a dedicated client with SDK retries disabled and a ten-minute
application retry budget. It retries model restoration and temporary failures
with backoff, while permissions, invalid configuration, and malformed responses
fail immediately. A single invocation may wait up to 310 seconds, bounded by
the remaining budget. The budget controls attempts; transport connection and
response reads can add a small amount of time to the final attempt. The
`uvicorn.error.bedrock_warmup` logger records duration, attempts, and outcomes.

This starts restoration earlier; a visitor who submits immediately may still
wait. Even a one-token warm-up is billable. Imported models are billed in
five-minute windows and scale down after five idle minutes. There are no
periodic keep-alive calls. See [AWS imported-model invocation](https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html)
and [pricing](https://aws.amazon.com/bedrock/pricing/).

#### Deploy on the existing EC2 host

From the repository root on EC2, install the supplied drop-in. It preserves
the existing `backend.service` command and its two Gunicorn workers:

```bash
sudo install -d -m 0755 /etc/systemd/system/backend.service.d
sudo install -m 0644 demo/deployment/backend.service.d/warmup.conf /etc/systemd/system/backend.service.d/warmup.conf
sudo systemctl daemon-reload
sudo systemctl restart backend.service
sudo systemctl status backend.service --no-pager
```

Systemd creates `/run/aspireai-backend`, owned by the service's existing
`User=jaaydin`, with mode `0700`. Do not set a different
`BEDROCK_WARMUP_STATE_DIR` in `.env`/`EnvironmentFile`; systemd's environment-file
values override `Environment=`. Local development needs no systemd changes and
uses a private directory under the OS temporary directory. Shared filesystem
coordination supports one host; use a shared coordinator before adding hosts.

Before deploying the frontend, compare these read-only checks:

```bash
curl -i http://127.0.0.1:3000/warmup
curl -i https://lighthouse.chat.library.sc.edu/api/warmup
```

Both must reach the backend and return the same status with `no-store`.
Check `sudo nginx -T` if `/api/warmup` returns a proxy 404 or the static page.
The existing `/api/` location should proxy all methods to
`http://127.0.0.1:3000/`, with the trailing slash stripping the `/api/` prefix.
If the configuration only forwards `/api/generate`, add a corresponding
`/api/warmup` route for both GET and POST. After any proxy edit, run
`sudo nginx -t` and `sudo systemctl reload nginx`.

Deploy the frontend using the existing script, which backs up `/var/www/html`:

```bash
bash demo/deploy_frontend.sh
sudo journalctl -u backend.service -f
```

For live acceptance, leave the model idle for more than five minutes with all
demo tabs closed. Open the demo and confirm `POST /api/warmup` precedes any
question; opening additional tabs must join the same job. Submit a question
while warming and confirm one user message, followed by one answer after ready.
The journal should show one warm-up job. Once ready, network activity must show
no further warm-up polling or keep-alive requests. Check failure/retry behavior
offline rather than changing production permissions.

Run all offline backend regression tests (no AWS calls) from `demo/backend`:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

The suite covers two processes sharing one invocation, worker death and restart,
readiness expiry, retry budgets/cooldowns, response validation, and API
responsiveness during inference. For browser regression tests, see
`../frontend/tests/test_warmup_browser.py`.

For a live M14 smoke test using your AWS credentials:

```bash
uv run python -m tests.test_bedrock
```

### Keyword retrieval (RAG option)

The existing `"model": "RAG"` API option now uses
`src.api.chatbots.rag.KeywordRetriever`, ported from
`evaluation/rag/retrievers/agentic.py`. It generates search terms with `M14`,
normalizes and deduplicates them, and ranks documents using BM25. `RAGChatbot`
passes the retrieved context and original question to `M14` for the answer
and returns the source titles, descriptions, and transcripts to the frontend.
The answer prompt matches the evaluation's `p1.yaml` and `c1.yaml` templates
(`Use the provided texts to answer the question` with `<item>` records), without
the additional instruction to respond that records are insufficient.
No frontend changes or embedding model downloads are required.

The retriever can also be used directly:

```python
from src.api.chatbots.rag import KeywordRetriever, RAGChatbot

# Use an existing backend chatbot instance for keyword generation and answers.
retriever = KeywordRetriever(database_url="dbname=lighthouse_rag_pages", chatbot=chatbot)
rag = RAGChatbot(id="keyword-rag", chatbot=chatbot, retriever=retriever)
```

The defaults retrieve three documents using at most eight keywords, a 64-token
keyword generation limit, the `search_text` column (title, description, and
transcript), and the `documents_search_text_bm25` index. Configure these with
`RAG_TOP_K`, `RAG_KEYWORD_COUNT`, `RAG_KEYWORD_MAX_NEW_TOKENS`,
`RAG_SEARCH_COLUMN`, and `RAG_INDEX_NAME`. `RAG_MAX_CONTEXT_CHARS` caps answer
context; the request's `max_new_tokens` applies to answer generation separately.
`RAG_SEARCH_FIELD` and `RAG_EMBEDDING_DEVICE` no longer configure the RAG option.

This requires the `pg_textsearch` extension, generated `search_text` column,
and BM25 index described in [the database setup instructions](db/README.md).
For an existing database, install the extension as documented there, then run
from `demo/backend` before using keyword retrieval:

```bash
psql -d lighthouse_rag_pages -v ON_ERROR_STOP=1 -f db/schema.sql
```

Restart the backend after changing its settings. The vector retriever remains
available as `PostgresContextRetriever` for direct use.

### SafeGenChat (SGC)

Run the supplied SafeGenChat service separately on the same host before using
the `SGC` model. The backend calls `http://127.0.0.1:8001/solve` by default:

```bash
SGC_PORT=8001 uv run python main.py
```

Change `configs/chatbots/sgc.yaml` if the SGC service uses another local port.





## Dev Notes

- gaico requires python version >=3.10,<3.13
- /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv
