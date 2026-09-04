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





## Dev Notes

- gaico requires python version >=3.10,<3.13
- /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv venv
