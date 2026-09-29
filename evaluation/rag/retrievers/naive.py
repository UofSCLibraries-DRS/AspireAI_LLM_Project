from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F

from chatbots.base import Chatbot


SearchField = Literal["title", "description", "transcript", "all"]
VECTOR_COLUMNS = {
    "title": "title_embedding",
    "description": "description_embedding",
    "transcript": "transcript_embedding",
}


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class E5QueryEmbedder:
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str = "cpu",
        embedding_size: int = 768,
        query_prefix: str = "query: ",
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        if tokenizer is None or model is None:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

        self.tokenizer = tokenizer
        self.model = model
        self.embedding_size = embedding_size
        self.query_prefix = query_prefix
        self.device = self._resolve_device(device)

        if int(model.config.hidden_size) != embedding_size:
            raise ValueError(
                f"Expected {embedding_size}-dimensional embeddings from {model_name}, "
                f"but the model reports {model.config.hidden_size}"
            )

        self.max_length = min(
            int(tokenizer.model_max_length),
            int(model.config.max_position_embeddings),
        )
        model.to(self.device)
        model.eval()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but it is unavailable")
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise ValueError("MPS was requested, but it is unavailable")
        return device

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("query must not be blank")

        tokenized = self.tokenizer(
            f"{self.query_prefix}{query}",
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device)
            for key, value in tokenized.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("The E5 tokenizer did not return an attention mask")

        with torch.inference_mode():
            outputs = self.model(**inputs)
            mask = attention_mask[..., None].bool()
            pooled = outputs.last_hidden_state.masked_fill(~mask, 0.0).sum(dim=1)
            pooled /= attention_mask.sum(dim=1)[..., None]
            embedding = F.normalize(pooled, p=2, dim=1)[0]

        values = embedding.to(device="cpu", dtype=torch.float32).tolist()
        if len(values) != self.embedding_size or not torch.isfinite(embedding).all():
            raise RuntimeError("The E5 model produced an invalid query embedding")
        return values


ConnectionFactory = Callable[[str], Any]


class NaiveRetriever:
    """Retrieve E5-ranked records from the pgvector schema used by the demo."""

    def __init__(
        self,
        database_url: str,
        embedding_model: str = "intfloat/e5-base-v2",
        embedding_device: str = "cpu",
        embedding_size: int = 768,
        search_field: SearchField = "all",
        query_prefix: str = "query: ",
        *,
        chatbot: Chatbot,
        embedder: QueryEmbedder | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if not database_url.strip():
            raise ValueError("database_url must not be blank")
        if search_field not in {*VECTOR_COLUMNS, "all"}:
            raise ValueError(
                "search_field must be title, description, transcript, or all"
            )

        self.database_url = database_url
        self.search_field = search_field
        self.chatbot = chatbot
        self.embedder = embedder or E5QueryEmbedder(
            model_name=embedding_model,
            device=embedding_device,
            embedding_size=embedding_size,
            query_prefix=query_prefix,
        )
        self.connection_factory = connection_factory or self._connect

    @staticmethod
    def _connect(database_url: str) -> Any:
        import psycopg

        return psycopg.connect(database_url)

    def retrieve(self, query: str, top_k: int) -> list[dict[str, str]]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        embedding = self.embedder.embed_query(query)
        vector = "[" + ",".join(format(value, ".9g") for value in embedding) + "]"
        sql, parameters = self._query(vector, top_k)

        with (
            self.connection_factory(self.database_url) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()

        return self._format_rows(rows, top_k)

    @classmethod
    def _format_rows(
        cls,
        rows: Sequence[Sequence[Any]],
        top_k: int,
    ) -> list[dict[str, str]]:
        results = []
        seen_ids = set()
        for document_id, collection, title, description, transcript, _ in rows:
            if document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            values = {
                "collection": str(collection).strip(),
                "title": title.strip() if title else "",
                "description": description.strip() if description else "",
                "transcript": transcript.strip() if transcript else "",
            }
            values["text"] = cls._format_record(values)
            results.append(values)
            if len(results) == top_k:
                break
        return results

    def _query(self, vector: str, top_k: int) -> tuple[str, Sequence[object]]:
        if self.search_field != "all":
            column = VECTOR_COLUMNS[self.search_field]
            return (
                f"""
SELECT id, collection, title, description, transcript,
       {column} <=> %s::vector AS distance
FROM documents
WHERE {column} IS NOT NULL
ORDER BY {column} <=> %s::vector
LIMIT %s
""",
                (vector, vector, top_k),
            )

        candidates = []
        parameters: list[object] = []
        for column in VECTOR_COLUMNS.values():
            candidates.append(
                f"""(
SELECT id, collection, title, description, transcript,
       {column} <=> %s::vector AS distance
FROM documents
WHERE {column} IS NOT NULL
ORDER BY {column} <=> %s::vector
LIMIT %s
)"""
            )
            parameters.extend((vector, vector, top_k))
        parameters.append(top_k)
        return (
            f"""
WITH candidates AS (
    {" UNION ALL ".join(candidates)}
), best_matches AS (
    SELECT DISTINCT ON (id)
           id, collection, title, description, transcript, distance
    FROM candidates
    ORDER BY id, distance
)
SELECT id, collection, title, description, transcript, distance
FROM best_matches
ORDER BY distance
LIMIT %s
""",
            tuple(parameters),
        )

    @staticmethod
    def _format_record(values: dict[str, str]) -> str:
        sections = [f"Collection: {values['collection']}"]
        if values["title"]:
            sections.append(f"Title: {values['title']}")
        if values["description"]:
            sections.append(f"Description: {values['description']}")
        if values["transcript"]:
            sections.append(f"Transcript:\n{values['transcript']}")
        return "\n".join(sections)
