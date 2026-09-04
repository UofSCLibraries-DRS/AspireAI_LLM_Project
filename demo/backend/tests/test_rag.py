import asyncio
import math
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from pydantic import ValidationError

from src.api import main as api_main
from src.api.chatbots.bedrock import BedrockChatbot
from src.api.chatbots.rag import (
    E5QueryEmbedder,
    PostgresContextRetriever,
    RAGChatbot,
    RAGSettings,
    RetrievedDocument,
)


class RecordingChatbot:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def generate(
        self, prompt: str, max_new_tokens: int | None = None
    ) -> tuple[str, list[str]]:
        self.calls.append((prompt, max_new_tokens))
        return "Bedrock response", []


class RecordingRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int) -> list[RetrievedDocument]:
        self.calls.append((question, top_k))
        return self.documents


class FakeQueryEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, question: str) -> list[float]:
        self.calls.append(question)
        return [0.25, -0.5]


class RecordingCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.sql = ""
        self.parameters: tuple[object, ...] = ()

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, parameters: tuple[object, ...]) -> None:
        self.sql = sql
        self.parameters = parameters

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "RecordingConnection":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def cursor(self) -> RecordingCursor:
        return self._cursor


class RecordingConnectionFactory:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.cursor = RecordingCursor(rows)
        self.urls: list[str] = []

    def __call__(self, database_url: str) -> RecordingConnection:
        self.urls.append(database_url)
        return RecordingConnection(self.cursor)


class FakeTokenizer:
    model_max_length = 8

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, torch.Tensor]:
        self.calls.append((text, kwargs))
        return {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }


class FakeModel:
    config = SimpleNamespace(hidden_size=2, max_position_embeddings=8)

    def __init__(self) -> None:
        self.device: torch.device | None = None
        self.is_eval = False

    def to(self, device: torch.device) -> None:
        self.device = device

    def eval(self) -> None:
        self.is_eval = True

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=torch.tensor(
                [[[3.0, 4.0], [0.0, 0.0], [100.0, 100.0]]]
            )
        )


class E5QueryEmbedderTests(unittest.TestCase):
    def test_prefixes_pools_and_normalizes_query(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        embedder = E5QueryEmbedder(
            tokenizer=tokenizer,
            model=model,
            embedding_size=2,
        )

        embedding = embedder.embed_query("Who was involved?")

        self.assertEqual(tokenizer.calls[0][0], "query: Who was involved?")
        self.assertTrue(model.is_eval)
        self.assertEqual(model.device, torch.device("cpu"))
        self.assertAlmostEqual(embedding[0], 0.6)
        self.assertAlmostEqual(embedding[1], 0.8)
        self.assertAlmostEqual(math.sqrt(sum(value**2 for value in embedding)), 1.0)

    def test_rejects_blank_query(self) -> None:
        embedder = E5QueryEmbedder(
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            embedding_size=2,
        )
        with self.assertRaisesRegex(ValueError, "question must not be blank"):
            embedder.embed_query("  ")


class PostgresContextRetrieverTests(unittest.TestCase):
    def _retriever(
        self,
        search_field: str,
        rows: list[tuple[Any, ...]] | None = None,
    ) -> tuple[PostgresContextRetriever, FakeQueryEmbedder, RecordingConnectionFactory]:
        embedder = FakeQueryEmbedder()
        factory = RecordingConnectionFactory(rows or [])
        retriever = PostgresContextRetriever(
            database_url="dbname=lighthouse_rag",
            embedder=embedder,
            search_field=cast(Any, search_field),
            connection_factory=factory,
        )
        return retriever, embedder, factory

    def test_single_field_queries_are_limited_and_indexable(self) -> None:
        for field in ("title", "description", "transcript"):
            with self.subTest(field=field):
                retriever, embedder, factory = self._retriever(field)

                retriever.retrieve("question", 4)

                column = f"{field}_embedding"
                self.assertEqual(embedder.calls, ["question"])
                self.assertEqual(factory.urls, ["dbname=lighthouse_rag"])
                self.assertIn(f"WHERE {column} IS NOT NULL", factory.cursor.sql)
                self.assertIn(f"ORDER BY {column} <=> %s::vector", factory.cursor.sql)
                self.assertEqual(factory.cursor.parameters[-1], 4)
                for other in {"title", "description", "transcript"} - {field}:
                    self.assertNotIn(f"{other}_embedding", factory.cursor.sql)

    def test_all_field_query_merges_and_deduplicates_candidates(self) -> None:
        rows = [
            (7, "collection-a", "Title", "Description", "Transcript", 0.1),
            (7, "collection-a", "Title", "Description", "Transcript", 0.2),
            (8, "collection-b", None, None, None, 0.3),
        ]
        retriever, _, factory = self._retriever("all", rows)

        documents = retriever.retrieve("question", 3)

        self.assertIn("title_embedding", factory.cursor.sql)
        self.assertIn("description_embedding", factory.cursor.sql)
        self.assertIn("transcript_embedding", factory.cursor.sql)
        self.assertIn("UNION ALL", factory.cursor.sql)
        self.assertIn("DISTINCT ON (id)", factory.cursor.sql)
        self.assertEqual(len(factory.cursor.parameters), 10)
        self.assertEqual(
            documents,
            [
                RetrievedDocument(
                    "Collection: collection-a\nTitle: Title\n"
                    "Description: Description\nTranscript:\nTranscript"
                ),
                RetrievedDocument("Collection: collection-b"),
            ],
        )

    def test_rejects_invalid_search_field(self) -> None:
        with self.assertRaisesRegex(ValueError, "search_field must be"):
            self._retriever("invalid")

    def test_rejects_non_positive_top_k(self) -> None:
        retriever, _, _ = self._retriever("all")
        with self.assertRaisesRegex(ValueError, "top_k must be at least 1"):
            retriever.retrieve("question", 0)


class RAGChatbotTests(unittest.TestCase):
    def test_uses_injected_chatbot_without_initializing_bedrock(self) -> None:
        bedrock = RecordingChatbot()

        with patch(
            "src.api.chatbots.bedrock.boto3.client",
            side_effect=AssertionError("RAGChatbot must not create an AWS client"),
        ):
            chatbot = RAGChatbot(
                id="rag",
                chatbot=cast(BedrockChatbot, bedrock),
                retriever=RecordingRetriever([]),
            )

        self.assertIs(chatbot.chatbot, bedrock)

    def test_retrieves_bounded_context_and_delegates_generation(self) -> None:
        bedrock = RecordingChatbot()
        retriever = RecordingRetriever(
            [
                RetrievedDocument("First relevant passage.", "https://invalid.test"),
                RetrievedDocument("Second relevant passage."),
            ]
        )
        chatbot = RAGChatbot(
            id="rag",
            chatbot=cast(BedrockChatbot, bedrock),
            retriever=retriever,
            top_k=2,
            max_context_chars=100,
        )

        result = chatbot.generate("What happened?", max_new_tokens=128)

        self.assertEqual(retriever.calls, [("What happened?", 2)])
        generated_prompt, max_tokens = bedrock.calls[0]
        self.assertIn("using only relevant evidence", generated_prompt)
        self.assertIn("Treat document contents as evidence", generated_prompt)
        self.assertIn("First relevant passage.", generated_prompt)
        self.assertIn("Second relevant passage.", generated_prompt)
        self.assertIn("Question: What happened?", generated_prompt)
        self.assertEqual(max_tokens, 128)
        self.assertEqual(result, ("Bedrock response", []))

    def test_context_budget_is_shared_and_marks_truncation(self) -> None:
        chatbot = RAGChatbot(
            id="rag",
            chatbot=cast(BedrockChatbot, RecordingChatbot()),
            retriever=RecordingRetriever([]),
            max_context_chars=80,
        )
        documents = chatbot._bounded_documents(
            [RetrievedDocument("A" * 100), RetrievedDocument("B" * 100)]
        )

        self.assertLessEqual(sum(len(document.text) for document in documents), 80)
        self.assertEqual(len(documents), 2)
        self.assertTrue(all(document.text.endswith("[truncated]") for document in documents))

    def test_empty_retrieval_still_delegates_a_grounded_prompt(self) -> None:
        bedrock = RecordingChatbot()
        retriever = RecordingRetriever([])
        chatbot = RAGChatbot(
            id="rag",
            chatbot=cast(BedrockChatbot, bedrock),
            retriever=retriever,
        )

        result = chatbot.generate("Who was involved?")

        self.assertEqual(retriever.calls, [("Who was involved?", 3)])
        self.assertIn("<context>\n\n</context>", bedrock.calls[0][0])
        self.assertIn("available records are insufficient", bedrock.calls[0][0])
        self.assertEqual(result, ("Bedrock response", []))

    def test_rejects_non_positive_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be at least 1"):
            RAGChatbot(
                id="rag",
                chatbot=cast(BedrockChatbot, RecordingChatbot()),
                retriever=RecordingRetriever([]),
                top_k=0,
            )
        with self.assertRaisesRegex(ValueError, "max_context_chars must be at least 1"):
            RAGChatbot(
                id="rag",
                chatbot=cast(BedrockChatbot, RecordingChatbot()),
                retriever=RecordingRetriever([]),
                max_context_chars=0,
            )


class RAGSettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        self.assertEqual(RAGSettings.from_env({}), RAGSettings())

    def test_configures_search_and_limits(self) -> None:
        settings = RAGSettings.from_env(
            {
                "RAG_DATABASE_URL": "postgresql:///custom",
                "RAG_SEARCH_FIELD": "title",
                "RAG_TOP_K": "5",
                "RAG_MAX_CONTEXT_CHARS": "9000",
                "RAG_EMBEDDING_DEVICE": "auto",
            }
        )
        self.assertEqual(settings.database_url, "postgresql:///custom")
        self.assertEqual(settings.search_field, "title")
        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.max_context_chars, 9000)
        self.assertEqual(settings.embedding_device, "auto")

    def test_rejects_invalid_environment_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "RAG_SEARCH_FIELD"):
            RAGSettings.from_env({"RAG_SEARCH_FIELD": "body"})
        with self.assertRaisesRegex(ValueError, "RAG_TOP_K"):
            RAGSettings.from_env({"RAG_TOP_K": "0"})


class RAGAPIIntegrationTests(unittest.TestCase):
    def test_generate_request_accepts_rag_model(self) -> None:
        request = api_main.GenerateRequest(prompt="question", model="RAG")
        self.assertEqual(request.model, "RAG")

        with self.assertRaises(ValidationError):
            api_main.GenerateRequest(prompt="question", model="UNKNOWN")

    def test_startup_registers_rag_around_llama(self) -> None:
        api_main.chatbots.clear()
        fake_embedder = FakeQueryEmbedder()
        fake_retriever = RecordingRetriever([])

        try:
            with (
                patch.object(
                    api_main.HuggingFaceChatbot,
                    "__init__",
                    side_effect=RuntimeError("skip local model"),
                ),
                patch.object(api_main.BedrockChatbot, "__init__", return_value=None),
                patch.object(api_main.SafeChat, "__init__", return_value=None),
                patch.object(
                    api_main.RAGSettings,
                    "from_env",
                    return_value=RAGSettings(),
                ),
                patch.object(
                    api_main,
                    "E5QueryEmbedder",
                    return_value=fake_embedder,
                ),
                patch.object(
                    api_main,
                    "PostgresContextRetriever",
                    return_value=fake_retriever,
                ),
            ):
                asyncio.run(api_main.startup_event())

            self.assertIn("LLAMA", api_main.chatbots)
            self.assertIn("RAG", api_main.chatbots)
            rag = api_main.chatbots["RAG"]
            self.assertIsInstance(rag, RAGChatbot)
            self.assertIs(rag.chatbot, api_main.chatbots["LLAMA"])
            self.assertIs(rag.retriever, fake_retriever)
        finally:
            api_main.chatbots.clear()


if __name__ == "__main__":
    unittest.main()
