import json
import re
from collections.abc import Sequence
from typing import Any

from chatbots.base import Chatbot
from retrievers.naive import (
    ConnectionFactory,
    NaiveRetriever,
    QueryEmbedder,
    SearchField,
)

DEFAULT_KEYWORD_PROMPT = """You are generating a BM25 search query for a document
retriever. Follow these instructions exactly:

- Select at most {keyword_count} concise keywords or short noun phrases.
- Prefer distinctive names, events, places, dates, and domain terminology.
- Return one comma-separated line and nothing else.
- Do not answer the question.
- Do not include commentary, labels, numbering, or Markdown.

<question>
{query}
</question>"""

DEFAULT_HYDE_ANSWER_PROMPT = """Generate a hypothetical answer for semantic document
retrieval. Follow these instructions exactly:

- Write one concise, self-contained answer in a factual reference style.
- Treat it as a plausible answer, even when the source facts are uncertain.
- Include names, events, places, dates, and terminology likely to occur in sources.
- Return only the answer text.
- Do not include a preamble, disclaimer, analysis, citations, or Markdown.

<question>
{query}
</question>"""

DEFAULT_HYDE_DOCUMENT_PROMPT = """Generate a hypothetical source passage for semantic
document retrieval. Follow these instructions exactly:

- Write one short passage that could plausibly appear in a relevant source document.
- State the information that would be useful for answering the question.
- Include names, events, places, dates, and terminology likely to occur in sources.
- Return only the passage text.
- Do not include a preamble, disclaimer, analysis, citations, or Markdown.

<question>
{query}
</question>"""

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_KEYWORD_PREFIX = re.compile(r"^\s*(?:search\s+)?keywords?\s*:\s*", re.IGNORECASE)
_LIST_PREFIX = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def parse_keywords(generation: str, max_keywords: int = 8) -> str:
    """Normalize common LLM list formats into pg_textsearch query text."""
    if max_keywords < 1:
        raise ValueError("max_keywords must be at least 1")
    if not isinstance(generation, str) or not generation.strip():
        raise ValueError("The chatbot generated no keywords")

    text = generation.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()

    candidates: list[str]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        parsed = parsed.get("keywords")
    if isinstance(parsed, list):
        candidates = [str(value) for value in parsed if isinstance(value, str)]
    else:
        text = _KEYWORD_PREFIX.sub("", text)
        candidates = re.split(r"[,;|\n]+", text)

    keywords = []
    seen = set()
    for candidate in candidates:
        keyword = _LIST_PREFIX.sub("", candidate).strip().strip("\"'`[]()")
        keyword = _KEYWORD_PREFIX.sub("", keyword).strip()
        if not keyword:
            continue
        normalized = keyword.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        keywords.append(keyword)
        if len(keywords) == max_keywords:
            break

    if not keywords:
        raise ValueError("The chatbot generated no usable keywords")
    return " ".join(keywords)


def _validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a simple PostgreSQL identifier")
    return value


def _validate_prompt_template(template: str, name: str) -> str:
    if not isinstance(template, str) or "{query}" not in template:
        raise ValueError(f"{name} must be a string containing {{query}}")
    return template


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _generate(chatbot: Chatbot, prompt: str, max_new_tokens: int) -> str:
    generation, _ = chatbot.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    if not isinstance(generation, str) or not generation.strip():
        raise ValueError("The chatbot generated an empty retrieval query")
    return generation.strip()


class KeywordRetriever:
    """Generate simple search terms with an LLM and rank documents with BM25."""

    def __init__(
        self,
        database_url: str,
        search_column: str = "search_text",
        index_name: str = "documents_search_text_bm25",
        keyword_count: int = 8,
        max_new_tokens: int = 64,
        prompt_template: str = DEFAULT_KEYWORD_PROMPT,
        *,
        chatbot: Chatbot,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must not be blank")
        self.database_url = database_url
        self.search_column = _validate_identifier(search_column, "search_column")
        self.index_name = _validate_identifier(index_name, "index_name")
        self.keyword_count = _validate_positive_int(keyword_count, "keyword_count")
        self.max_new_tokens = _validate_positive_int(max_new_tokens, "max_new_tokens")
        self.prompt_template = _validate_prompt_template(
            prompt_template,
            "prompt_template",
        )
        self.chatbot = chatbot
        self.connection_factory = connection_factory or NaiveRetriever._connect

    def generate_keywords(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        prompt = self.prompt_template.format(
            query=query,
            keyword_count=self.keyword_count,
        )
        generation = _generate(self.chatbot, prompt, self.max_new_tokens)
        return parse_keywords(generation, self.keyword_count)

    def retrieve(self, query: str, top_k: int) -> list[dict[str, str]]:
        _validate_positive_int(top_k, "top_k")
        keywords = self.generate_keywords(query)
        sql, parameters = self._query(keywords, top_k)

        with (
            self.connection_factory(self.database_url) as connection,
            connection.cursor() as cursor,
        ):
            cursor.execute(sql, parameters)
            rows = cursor.fetchall()
        return NaiveRetriever._format_rows(rows, top_k)

    def _query(self, keywords: str, top_k: int) -> tuple[str, Sequence[object]]:
        # Identifiers are validated before interpolation. Query text remains a
        # bound value so generated content can never alter the SQL statement.
        score = f"{self.search_column} <@> to_bm25query(%s, '{self.index_name}')"
        return (
            f"""
SELECT id, collection, title, description, transcript,
       {score} AS score
FROM documents
ORDER BY {score}
LIMIT %s
""",
            (keywords, keywords, top_k),
        )


class HybridRetriever(NaiveRetriever):
    """Fuse vector and LLM-keyword BM25 rankings with reciprocal-rank fusion."""

    def __init__(
        self,
        database_url: str,
        embedding_model: str = "intfloat/e5-base-v2",
        embedding_device: str = "cpu",
        embedding_size: int = 768,
        search_field: SearchField = "all",
        query_prefix: str = "query: ",
        search_column: str = "search_text",
        index_name: str = "documents_search_text_bm25",
        keyword_count: int = 8,
        keyword_max_new_tokens: int = 64,
        keyword_prompt_template: str = DEFAULT_KEYWORD_PROMPT,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
        *,
        chatbot: Chatbot,
        embedder: QueryEmbedder | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        super().__init__(
            database_url=database_url,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            embedding_size=embedding_size,
            search_field=search_field,
            query_prefix=query_prefix,
            chatbot=chatbot,
            embedder=embedder,
            connection_factory=connection_factory,
        )
        self.keyword_retriever = KeywordRetriever(
            database_url=database_url,
            search_column=search_column,
            index_name=index_name,
            keyword_count=keyword_count,
            max_new_tokens=keyword_max_new_tokens,
            prompt_template=keyword_prompt_template,
            chatbot=chatbot,
            connection_factory=self.connection_factory,
        )
        self.rrf_k = _validate_positive_int(rrf_k, "rrf_k")
        self.candidate_multiplier = _validate_positive_int(
            candidate_multiplier,
            "candidate_multiplier",
        )

    def retrieve(self, query: str, top_k: int) -> list[dict[str, str]]:
        _validate_positive_int(top_k, "top_k")
        keywords = self.keyword_retriever.generate_keywords(query)
        vector = self.embedder.embed_query(query)
        vector_literal = "[" + ",".join(format(value, ".9g") for value in vector) + "]"
        candidate_k = top_k * self.candidate_multiplier
        vector_sql, vector_parameters = self._query(vector_literal, candidate_k)
        keyword_sql, keyword_parameters = self.keyword_retriever._query(
            keywords,
            candidate_k,
        )

        with self.connection_factory(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(vector_sql, vector_parameters)
                vector_rows = cursor.fetchall()
            with connection.cursor() as cursor:
                cursor.execute(keyword_sql, keyword_parameters)
                keyword_rows = cursor.fetchall()

        return self._fuse(vector_rows, keyword_rows, top_k)

    def _fuse(
        self,
        vector_rows: Sequence[Sequence[Any]],
        keyword_rows: Sequence[Sequence[Any]],
        top_k: int,
    ) -> list[dict[str, str]]:
        documents: dict[Any, Sequence[Any]] = {}
        scores: dict[Any, float] = {}
        first_seen: dict[Any, int] = {}

        for rows in (vector_rows, keyword_rows):
            seen_in_ranking = set()
            rank = 0
            for row in rows:
                document_id = row[0]
                if document_id in seen_in_ranking:
                    continue
                seen_in_ranking.add(document_id)
                rank += 1
                documents.setdefault(document_id, row)
                first_seen.setdefault(document_id, len(first_seen))
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )

        ordered_ids = sorted(
            documents,
            key=lambda document_id: (-scores[document_id], first_seen[document_id]),
        )
        fused_rows = [
            (*documents[document_id][:5], scores[document_id])
            for document_id in ordered_ids[:top_k]
        ]
        return self._format_rows(fused_rows, top_k)


class _HyDERetriever(NaiveRetriever):
    prompt_name = "prompt_template"

    def __init__(
        self,
        database_url: str,
        embedding_model: str = "intfloat/e5-base-v2",
        embedding_device: str = "cpu",
        embedding_size: int = 768,
        search_field: SearchField = "all",
        query_prefix: str = "passage: ",
        max_new_tokens: int = 256,
        prompt_template: str = "{query}",
        *,
        chatbot: Chatbot,
        embedder: QueryEmbedder | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        super().__init__(
            database_url=database_url,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            embedding_size=embedding_size,
            search_field=search_field,
            query_prefix=query_prefix,
            chatbot=chatbot,
            embedder=embedder,
            connection_factory=connection_factory,
        )
        self.max_new_tokens = _validate_positive_int(max_new_tokens, "max_new_tokens")
        self.prompt_template = _validate_prompt_template(
            prompt_template,
            self.prompt_name,
        )

    def retrieve(self, query: str, top_k: int) -> list[dict[str, str]]:
        _validate_positive_int(top_k, "top_k")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        prompt = self.prompt_template.format(query=query)
        hypothetical_text = _generate(self.chatbot, prompt, self.max_new_tokens)
        return super().retrieve(hypothetical_text, top_k)


class HyDEAnswerRetriever(_HyDERetriever):
    """Generate and embed a hypothetical answer before vector retrieval."""

    def __init__(
        self,
        *args: Any,
        prompt_template: str = DEFAULT_HYDE_ANSWER_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, prompt_template=prompt_template, **kwargs)


class HyDEDocumentRetriever(_HyDERetriever):
    """Generate and embed a hypothetical source passage before vector retrieval."""

    def __init__(
        self,
        *args: Any,
        prompt_template: str = DEFAULT_HYDE_DOCUMENT_PROMPT,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, prompt_template=prompt_template, **kwargs)
