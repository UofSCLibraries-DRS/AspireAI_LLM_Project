import unittest
from typing import cast
from unittest.mock import patch

from src.api.chatbots.bedrock import BedrockChatbot
from src.api.chatbots.rag import RAGChatbot, RetrievedDocument


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

    def test_retrieves_context_and_delegates_generation(self) -> None:
        bedrock = RecordingChatbot()
        retriever = RecordingRetriever(
            [
                RetrievedDocument(
                    text="First relevant passage.",
                    source="https://example.com/first",
                ),
                RetrievedDocument(text="Second relevant passage."),
                RetrievedDocument(
                    text="Third relevant passage.",
                    source=" https://example.com/third ",
                ),
            ]
        )
        chatbot = RAGChatbot(
            id="rag",
            chatbot=cast(BedrockChatbot, bedrock),
            retriever=retriever,
            top_k=3,
        )

        result = chatbot.generate("What happened?", max_new_tokens=128)

        self.assertEqual(retriever.calls, [("What happened?", 3)])
        self.assertEqual(
            bedrock.calls,
            [
                (
                    "Use the following context to answer the question.\n\n"
                    "<context>\n"
                    "<document>\nFirst relevant passage.\n</document>\n\n"
                    "<document>\nSecond relevant passage.\n</document>\n\n"
                    "<document>\nThird relevant passage.\n</document>\n"
                    "</context>\n\n"
                    "Question: What happened?",
                    128,
                )
            ],
        )
        self.assertEqual(
            result,
            (
                "Bedrock response",
                ["https://example.com/first", "https://example.com/third"],
            ),
        )

    def test_empty_retrieval_still_delegates_a_valid_prompt(self) -> None:
        bedrock = RecordingChatbot()
        retriever = RecordingRetriever([])
        chatbot = RAGChatbot(
            id="rag",
            chatbot=cast(BedrockChatbot, bedrock),
            retriever=retriever,
        )

        result = chatbot.generate("Who was involved?")

        self.assertEqual(retriever.calls, [("Who was involved?", 1)])
        self.assertEqual(
            bedrock.calls,
            [
                (
                    "Use the following context to answer the question.\n\n"
                    "<context>\n\n</context>\n\n"
                    "Question: Who was involved?",
                    None,
                )
            ],
        )
        self.assertEqual(result, ("Bedrock response", []))

    def test_rejects_non_positive_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k must be at least 1"):
            RAGChatbot(
                id="rag",
                chatbot=cast(BedrockChatbot, RecordingChatbot()),
                retriever=RecordingRetriever([]),
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()
