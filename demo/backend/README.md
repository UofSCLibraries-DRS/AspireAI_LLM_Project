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
