import unittest
from collections.abc import Sequence
from typing import Any

from retrievers.agentic import (
    HyDEAnswerRetriever,
    HyDEDocumentRetriever,
    HybridRetriever,
    KeywordRetriever,
    parse_keywords,
)
from retrievers.factory import RETRIEVER_BACKENDS, create_retriever
from utils.config import RetrieverSpec


class RecordingChatbot:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, int | None]] = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None,
    ) -> tuple[str, list[str]]:
        self.calls.append((prompt, max_new_tokens))
        return self.responses.pop(0), []


class RecordingEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.25, -0.5]


class ScriptedCursor:
    def __init__(
        self,
        rows: Sequence[Sequence[Any]],
        executions: list[tuple[str, Sequence[object]]],
    ) -> None:
        self.rows = rows
        self.executions = executions

    def __enter__(self) -> "ScriptedCursor":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def execute(self, sql: str, parameters: Sequence[object]) -> None:
        self.executions.append((sql, parameters))

    def fetchall(self) -> Sequence[Sequence[Any]]:
        return self.rows


class ScriptedConnection:
    def __init__(
        self,
        result_sets: list[Sequence[Sequence[Any]]],
        executions: list[tuple[str, Sequence[object]]],
    ) -> None:
        self.result_sets = result_sets
        self.executions = executions

    def __enter__(self) -> "ScriptedConnection":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def cursor(self) -> ScriptedCursor:
        return ScriptedCursor(self.result_sets.pop(0), self.executions)


class ScriptedConnectionFactory:
    def __init__(self, *result_sets: Sequence[Sequence[Any]]) -> None:
        self.result_sets = list(result_sets)
        self.executions: list[tuple[str, Sequence[object]]] = []
        self.urls: list[str] = []

    def __call__(self, database_url: str) -> ScriptedConnection:
        self.urls.append(database_url)
        return ScriptedConnection(self.result_sets, self.executions)


class ParseKeywordsTests(unittest.TestCase):
    def test_parses_json_and_deduplicates_case_insensitively(self) -> None:
        self.assertEqual(
            parse_keywords('["Voting rights", "Selma", "voting RIGHTS"]'),
            "Voting rights Selma",
        )

    def test_parses_markdown_list_and_honors_limit(self) -> None:
        self.assertEqual(
            parse_keywords("Keywords:\n- civil rights\n- Alabama\n- march", 2),
            "civil rights Alabama",
        )

    def test_rejects_empty_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "no keywords"):
            parse_keywords("   ")


class KeywordRetrieverTests(unittest.TestCase):
    def test_generates_keywords_and_uses_explicit_bm25_index(self) -> None:
        chatbot = RecordingChatbot("Keywords: voting rights, Selma, 1965")
        connections = ScriptedConnectionFactory(
            [(7, "archive", "Voting Rights", "Description", "Transcript", -2.5)]
        )
        retriever = KeywordRetriever(
            database_url="dbname=lighthouse_rag",
            chatbot=chatbot,  # type: ignore[arg-type]
            connection_factory=connections,
        )

        results = retriever.retrieve("What happened in Selma?", 3)

        prompt = chatbot.calls[0][0]
        self.assertTrue(prompt.startswith("You are generating a BM25 search query"))
        self.assertIn("Follow these instructions exactly", prompt)
        self.assertIn("<question>\nWhat happened in Selma?\n</question>", prompt)
        self.assertIn("Return one comma-separated line and nothing else", prompt)
        self.assertEqual(chatbot.calls[0][1], 64)
        sql, parameters = connections.executions[0]
        self.assertIn("search_text <@> to_bm25query", sql)
        self.assertIn("documents_search_text_bm25", sql)
        self.assertEqual(
            parameters,
            ("voting rights Selma 1965", "voting rights Selma 1965", 3),
        )
        self.assertEqual(results[0]["title"], "Voting Rights")
        self.assertIn("Transcript:\nTranscript", results[0]["text"])

    def test_rejects_unsafe_sql_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "simple PostgreSQL identifier"):
            KeywordRetriever(
                database_url="dbname=lighthouse_rag",
                index_name="index'); DROP TABLE documents; --",
                chatbot=RecordingChatbot("term"),  # type: ignore[arg-type]
            )


class HybridRetrieverTests(unittest.TestCase):
    def test_rrf_fuses_vector_and_keyword_rankings(self) -> None:
        chatbot = RecordingChatbot("civil rights, voting")
        embedder = RecordingEmbedder()
        connections = ScriptedConnectionFactory(
            [
                (1, "archive", "Vector first", "", "", 0.1),
                (2, "archive", "In both", "", "", 0.2),
            ],
            [
                (2, "archive", "In both", "", "", -3.0),
                (3, "archive", "Keyword second", "", "", -2.0),
            ],
        )
        retriever = HybridRetriever(
            database_url="dbname=lighthouse_rag",
            chatbot=chatbot,  # type: ignore[arg-type]
            embedder=embedder,
            connection_factory=connections,
            candidate_multiplier=4,
        )

        results = retriever.retrieve("Who could vote?", 3)

        self.assertEqual(embedder.queries, ["Who could vote?"])
        self.assertEqual(
            [result["title"] for result in results],
            ["In both", "Vector first", "Keyword second"],
        )
        self.assertEqual(len(connections.executions), 2)
        self.assertIn("title_embedding", connections.executions[0][0])
        self.assertIn("to_bm25query", connections.executions[1][0])
        self.assertEqual(connections.executions[1][1][-1], 12)


class HyDERetrieverTests(unittest.TestCase):
    def test_answer_embeds_generated_answer(self) -> None:
        chatbot = RecordingChatbot("A hypothetical answer with source terminology.")
        embedder = RecordingEmbedder()
        connections = ScriptedConnectionFactory([])
        retriever = HyDEAnswerRetriever(
            database_url="dbname=lighthouse_rag",
            search_field="title",
            chatbot=chatbot,  # type: ignore[arg-type]
            embedder=embedder,
            connection_factory=connections,
        )

        retriever.retrieve("What happened?", 2)

        prompt = chatbot.calls[0][0]
        self.assertTrue(prompt.startswith("Generate a hypothetical answer"))
        self.assertIn("Follow these instructions exactly", prompt)
        self.assertIn("<question>\nWhat happened?\n</question>", prompt)
        self.assertEqual(chatbot.calls[0][1], 256)
        self.assertEqual(
            embedder.queries,
            ["A hypothetical answer with source terminology."],
        )

    def test_document_embeds_generated_document(self) -> None:
        chatbot = RecordingChatbot("A hypothetical archival document passage.")
        embedder = RecordingEmbedder()
        connections = ScriptedConnectionFactory([])
        retriever = HyDEDocumentRetriever(
            database_url="dbname=lighthouse_rag",
            search_field="description",
            chatbot=chatbot,  # type: ignore[arg-type]
            embedder=embedder,
            connection_factory=connections,
        )

        retriever.retrieve("Which event?", 1)

        prompt = chatbot.calls[0][0]
        self.assertTrue(prompt.startswith("Generate a hypothetical source passage"))
        self.assertIn("Follow these instructions exactly", prompt)
        self.assertIn("<question>\nWhich event?\n</question>", prompt)
        self.assertEqual(embedder.queries, ["A hypothetical archival document passage."])

    def test_factory_registers_all_new_backends(self) -> None:
        self.assertEqual(
            {
                "HyDEAnswerRetriever",
                "HyDEDocumentRetriever",
                "HybridRetriever",
                "KeywordRetriever",
            },
            set(RETRIEVER_BACKENDS) - {"NaiveRetriever"},
        )

        chatbot = RecordingChatbot("keywords")
        retriever = create_retriever(
            RetrieverSpec(
                id="keyword",
                backend="KeywordRetriever",
                config={
                    "database_url": "dbname=lighthouse_rag",
                    "connection_factory": ScriptedConnectionFactory([]),
                },
            ),
            chatbot=chatbot,  # type: ignore[arg-type]
        )

        self.assertIsInstance(retriever, KeywordRetriever)
        self.assertIs(retriever.chatbot, chatbot)


if __name__ == "__main__":
    unittest.main()
