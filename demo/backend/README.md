## How to Run

### 1. Create virtual environment & install dependencies

```bash
uv sync
```

### 2. Start API server 
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The `RAG` model uses the local `lighthouse_rag` PostgreSQL database and wraps
the `LLAMA` Bedrock model. Its optional environment settings are documented in
`.env.example`; retrieval searches all stored embedding fields by default.

### Keyword retrieval (backend only)

`src.api.chatbots.rag.KeywordRetriever` generates search terms with an injected
backend `Chatbot` and ranks documents using BM25. It returns `RetrievedDocument`
objects with source content and can be passed directly to `RAGChatbot`:

```python
from src.api.chatbots.rag import KeywordRetriever, RAGChatbot

# Use an existing backend chatbot instance for keyword generation and answers.
retriever = KeywordRetriever(database_url="dbname=lighthouse_rag", chatbot=chatbot)
rag = RAGChatbot(id="keyword-rag", chatbot=chatbot, retriever=retriever)
```

The defaults use at most eight keywords, a 64-token generation limit, the
`search_text` column, and the `documents_search_text_bm25` index. Override them
with `keyword_count`, `max_new_tokens`, `search_column`, and `index_name`;
`prompt_template` must contain `{query}` and may also contain `{keyword_count}`.
This requires the `pg_textsearch` extension and BM25 schema described in
[the database setup instructions](db/README.md). API model selection and startup
continue to use the existing vector retriever.

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
