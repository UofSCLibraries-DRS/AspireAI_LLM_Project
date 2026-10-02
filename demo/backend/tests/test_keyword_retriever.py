import unittest
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

from src.api.chatbots.base import Chatbot
from src.api.chatbots.rag import (
    KeywordRetriever,
    RAGChatbot,
    RetrievedDocument,
    parse_keywords,
)


class ParseKeywordsTests(unittest.TestCase):
    def test_normalizes_common_formats_and_deduplicates(self) -> None:
        generations = [
            "Keywords: Voting rights, Selma, voting RIGHTS",
            "Search keywords:\n1. Voting rights\n2. Selma\n3. voting RIGHTS",
            "- Voting rights\n* Selma\n• voting RIGHTS",
            '["Voting rights", "Selma", "voting RIGHTS", 42, null]',
            '{"keywords": ["Voting rights", "Selma", "voting RIGHTS"]}',
            '```json\n["Voting rights", "Selma"]\n```',
            '"Voting rights"; `Selma` | (voting RIGHTS)',
        ]
        for generation in generations:
            with self.subTest(generation=generation):
                self.assertEqual(parse_keywords(generation), "Voting rights Selma")

    def test_limits_unique_keywords(self) -> None:
        self.assertEqual(
            parse_keywords("Selma, selma, voting, march", 2), "Selma voting"
        )

    def test_rejects_empty_or_unusable_generations(self) -> None:
        for generation in (
            "",
            "  ",
            "Keywords:",
            "[]",
            "[null, 42]",
            ",;|",
            "```\n```",
        ):
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                parse_keywords(generation)

    def test_rejects_invalid_limits(self) -> None:
        for limit in (0, -1, True, 1.5, "2", None):
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "max_keywords"),
            ):
                parse_keywords("term", cast(Any, limit))


class KeywordRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chatbot = Mock(spec=Chatbot)
        self.chatbot.generate.return_value = (
            "Keywords: voting rights, Selma, 1965",
            [],
        )
        self.connections = MagicMock()
        self.connection = self.connections.return_value.__enter__.return_value
        self.cursor = self.connection.cursor.return_value.__enter__.return_value
        self.cursor.fetchall.return_value = []

    def retriever(self, **kwargs: Any) -> KeywordRetriever:
        return KeywordRetriever(
            database_url="dbname=lighthouse_rag",
            chatbot=self.chatbot,
            connection_factory=self.connections,
            **kwargs,
        )

    def test_generates_keywords_and_returns_ranked_context_and_sources(self) -> None:
        self.cursor.fetchall.return_value = [
            (7, "archive", " Voting Rights ", " Description ", " Transcript ", -2.5),
            (7, "archive", "Duplicate", None, None, -2.0),
            (8, "other archive", None, "  ", None, -1.5),
            (9, "archive", "Beyond limit", None, None, -1.0),
        ]

        documents = self.retriever().retrieve(
            question="What happened in Selma?", top_k=2
        )

        prompt = self.chatbot.generate.call_args.kwargs["prompt"]
        self.assertIn("at most 8", prompt)
        self.assertIn("<question>\nWhat happened in Selma?\n</question>", prompt)
        self.assertEqual(self.chatbot.generate.call_args.kwargs["max_new_tokens"], 64)
        self.connections.assert_called_once_with("dbname=lighthouse_rag")
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn(
            "search_text <@> to_bm25query(%s, 'documents_search_text_bm25')", sql
        )
        self.assertIn("ORDER BY search_text <@>", sql)
        self.assertIn("LIMIT %s", sql)
        self.assertEqual(
            parameters, ("voting rights Selma 1965", "voting rights Selma 1965", 2)
        )
        self.assertEqual(
            documents,
            [
                RetrievedDocument(
                    "Collection: archive\nTitle: Voting Rights\nDescription: Description\nTranscript:\nTranscript",
                    {
                        "title": "Voting Rights",
                        "description": "Description",
                        "transcript": "Transcript",
                    },
                ),
                RetrievedDocument(
                    "Collection: other archive",
                    {"title": "other archive", "description": None, "transcript": ""},
                ),
            ],
        )
        self.connection.cursor.return_value.__exit__.assert_called_once()
        self.connections.return_value.__exit__.assert_called_once()

    def test_custom_prompt_limits_and_index(self) -> None:
        retriever = self.retriever(
            search_column="title",
            index_name="titles_bm25",
            keyword_count=2,
            max_new_tokens=32,
            prompt_template="Select {keyword_count} terms: {query}",
        )
        self.assertEqual(retriever.retrieve("question", 3), [])
        self.chatbot.generate.assert_called_once_with(
            prompt="Select 2 terms: question",
            max_new_tokens=32,
        )
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("title <@> to_bm25query(%s, 'titles_bm25')", sql)
        self.assertEqual(parameters, ("voting rights Selma", "voting rights Selma", 3))

    def test_generated_sql_like_text_is_only_a_bound_value(self) -> None:
        self.chatbot.generate.return_value = ("O'Brien DROP TABLE documents --", [])
        self.retriever().retrieve("question", 1)
        sql, parameters = self.cursor.execute.call_args.args
        self.assertNotIn("O'Brien", sql)
        self.assertNotIn("DROP TABLE", sql)
        self.assertEqual(parameters, ("O'Brien DROP TABLE documents --",) * 2 + (1,))

    def test_invalid_configuration_fails_before_generation_or_connection(self) -> None:
        invalid = [
            {"search_column": "search_text; DROP TABLE documents"},
            {"index_name": "index'); DROP TABLE documents; --"},
            {"prompt_template": "no query placeholder"},
            {"prompt_template": None},
        ]
        for name in ("keyword_count", "max_new_tokens"):
            invalid.extend({name: value} for value in (0, -1, True, 1.5, "2", None))
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.retriever(**kwargs)
        for database_url in ("", "  ", None):
            with (
                self.subTest(database_url=database_url),
                self.assertRaisesRegex(ValueError, "database_url"),
            ):
                KeywordRetriever(
                    database_url=cast(Any, database_url), chatbot=self.chatbot
                )
        self.chatbot.generate.assert_not_called()
        self.connections.assert_not_called()

    def test_invalid_request_fails_before_generation_or_connection(self) -> None:
        retriever = self.retriever()
        for question in ("", "  ", None):
            with self.subTest(question=question), self.assertRaises(ValueError):
                retriever.retrieve(cast(Any, question), 1)
        for top_k in (0, -1, True, 1.5, "2", None):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(ValueError, "top_k"):
                retriever.retrieve("question", cast(Any, top_k))
        self.chatbot.generate.assert_not_called()
        self.connections.assert_not_called()

    def test_empty_keywords_fail_before_connection(self) -> None:
        for generation in ("", "  ", None, "Keywords:", "[]"):
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                self.chatbot.generate.return_value = (generation, [])
                self.retriever().retrieve("question", 1)
        self.connections.assert_not_called()

    def test_generation_failure_propagates_without_querying_database(self) -> None:
        self.chatbot.generate.side_effect = RuntimeError("generation failed")
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.retriever().retrieve("question", 1)
        self.connections.assert_not_called()

    def test_database_failure_closes_resources(self) -> None:
        self.cursor.execute.side_effect = RuntimeError("query failed")
        with self.assertRaisesRegex(RuntimeError, "query failed"):
            self.retriever().retrieve("question", 1)
        self.connection.cursor.return_value.__exit__.assert_called_once()
        self.connections.return_value.__exit__.assert_called_once()

    def test_default_connection_uses_psycopg(self) -> None:
        with patch("psycopg.connect", self.connections):
            retriever = KeywordRetriever(
                database_url="dbname=lighthouse_rag", chatbot=self.chatbot
            )
            self.assertEqual(retriever.retrieve("question", 1), [])
        self.connections.assert_called_once_with("dbname=lighthouse_rag")

    def test_can_be_injected_into_rag_chatbot(self) -> None:
        self.cursor.fetchall.return_value = [
            (7, "archive", "Selma", None, "Evidence", -2.5)
        ]
        answer_chatbot = Mock()
        answer_chatbot.generate.return_value = ("Grounded answer", [])
        rag = RAGChatbot(
            id="keyword-rag", chatbot=answer_chatbot, retriever=self.retriever()
        )

        answer, sources = rag.generate("What happened in Selma?", max_new_tokens=128)

        self.assertEqual(answer, "Grounded answer")
        self.assertEqual(
            sources, [{"title": "Selma", "description": None, "transcript": "Evidence"}]
        )
        answer_prompt = answer_chatbot.generate.call_args.kwargs["prompt"]
        self.assertIn("Transcript:\nEvidence", answer_prompt)
        self.assertIn("Question: What happened in Selma?", answer_prompt)
        self.assertEqual(
            answer_chatbot.generate.call_args.kwargs["max_new_tokens"], 128
        )


if __name__ == "__main__":
    unittest.main()
