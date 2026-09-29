import math
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from retrievers.factory import create_retriever
from retrievers.naive import E5QueryEmbedder, NaiveRetriever
from utils.config import RetrieverSpec


class FakeTokenizer:
    model_max_length = 8

    def __init__(self) -> None:
        self.text = ""

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        self.text = text
        return {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }


class FakeModel:
    config = SimpleNamespace(hidden_size=2, max_position_embeddings=8)

    def to(self, _: torch.device) -> None:
        pass

    def eval(self) -> None:
        pass

    def __call__(self, **_: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            last_hidden_state=torch.tensor(
                [[[3.0, 4.0], [0.0, 0.0], [100.0, 100.0]]]
            )
        )


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.25, -0.5]


class FakeCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.sql = ""
        self.parameters: tuple[object, ...] = ()

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def execute(self, sql: str, parameters: tuple[object, ...]) -> None:
        self.sql = sql
        self.parameters = parameters

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def cursor(self) -> FakeCursor:
        return self._cursor


class FakeConnectionFactory:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.cursor = FakeCursor(rows)

    def __call__(self, _: str) -> FakeConnection:
        return FakeConnection(self.cursor)


class FakeAgenticRetriever:
    def __init__(self, chatbot: object, marker: str) -> None:
        self.chatbot = chatbot
        self.marker = marker


class E5QueryEmbedderTests(unittest.TestCase):
    def test_prefixes_mean_pools_and_normalizes(self) -> None:
        tokenizer = FakeTokenizer()
        embedder = E5QueryEmbedder(
            embedding_size=2,
            tokenizer=tokenizer,
            model=FakeModel(),
        )

        embedding = embedder.embed_query("Who was involved?")

        self.assertEqual(tokenizer.text, "query: Who was involved?")
        self.assertAlmostEqual(embedding[0], 0.6)
        self.assertAlmostEqual(embedding[1], 0.8)
        self.assertAlmostEqual(math.sqrt(sum(value**2 for value in embedding)), 1.0)


class NaiveRetrieverTests(unittest.TestCase):
    def _retriever(
        self,
        search_field: str = "all",
        rows: list[tuple[Any, ...]] | None = None,
    ) -> tuple[NaiveRetriever, FakeEmbedder, FakeConnectionFactory]:
        embedder = FakeEmbedder()
        connections = FakeConnectionFactory(rows or [])
        retriever = NaiveRetriever(
            database_url="dbname=lighthouse_rag",
            search_field=search_field,  # type: ignore[arg-type]
            chatbot=object(),  # type: ignore[arg-type]
            embedder=embedder,
            connection_factory=connections,
        )
        return retriever, embedder, connections

    def test_all_searches_each_field_and_formats_unique_records(self) -> None:
        retriever, embedder, connections = self._retriever(
            rows=[
                (7, "archive", "Title", "Description", "Transcript", 0.1),
                (7, "archive", "Title", "Description", "Transcript", 0.2),
                (8, "photos", None, None, None, 0.3),
            ]
        )

        results = retriever.retrieve("question", 3)

        self.assertEqual(embedder.queries, ["question"])
        self.assertIn("title_embedding", connections.cursor.sql)
        self.assertIn("description_embedding", connections.cursor.sql)
        self.assertIn("transcript_embedding", connections.cursor.sql)
        self.assertIn("DISTINCT ON (id)", connections.cursor.sql)
        self.assertEqual(len(connections.cursor.parameters), 10)
        self.assertEqual(
            [result["text"] for result in results],
            [
                "Collection: archive\nTitle: Title\nDescription: Description\n"
                "Transcript:\nTranscript",
                "Collection: photos",
            ],
        )

    def test_single_field_uses_its_vector_column(self) -> None:
        retriever, _, connections = self._retriever("title")

        retriever.retrieve("question", 4)

        self.assertIn("ORDER BY title_embedding <=> %s::vector", connections.cursor.sql)
        self.assertNotIn("description_embedding", connections.cursor.sql)
        self.assertEqual(connections.cursor.parameters, ("[0.25,-0.5]", "[0.25,-0.5]", 4))

    def test_factory_rejects_unknown_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown retriever backend"):
            create_retriever(
                RetrieverSpec("unknown", "Unknown", {}),
                chatbot=object(),  # type: ignore[arg-type]
            )

    def test_factory_injects_the_target_chatbot(self) -> None:
        chatbot = object()
        with patch.dict(
            "retrievers.factory.RETRIEVER_BACKENDS",
            {"AgenticRetriever": FakeAgenticRetriever},
        ):
            retriever = create_retriever(
                RetrieverSpec(
                    id="agentic",
                    backend="AgenticRetriever",
                    config={"marker": "configured"},
                ),
                chatbot=chatbot,  # type: ignore[arg-type]
            )

        self.assertIs(retriever.chatbot, chatbot)
        self.assertEqual(retriever.marker, "configured")


if __name__ == "__main__":
    unittest.main()
