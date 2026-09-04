from dataclasses import dataclass
from typing import Protocol

from .base import Chatbot
from .bedrock import BedrockChatbot


@dataclass(frozen=True)
class RetrievedDocument:
    """A context snippet and its optional public source URL."""

    text: str
    source: str | None = None


class ContextRetriever(Protocol):
    """Retrieval boundary for the future PostgreSQL implementation."""

    def retrieve(self, question: str, top_k: int) -> list[RetrievedDocument]:
        """Return the most relevant context snippets for a question."""
        ...


class RAGChatbot(Chatbot):
    """Add retrieved context to a question before delegating to Bedrock."""

    def __init__(
        self,
        id: str,
        chatbot: BedrockChatbot,
        retriever: ContextRetriever,
        top_k: int = 1,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        super().__init__(id=id)
        self.chatbot = chatbot
        self.retriever = retriever
        self.top_k = top_k

    @staticmethod
    def _augment_prompt(
        question: str, documents: list[RetrievedDocument]
    ) -> str:
        context = "\n\n".join(
            f"<document>\n{document.text}\n</document>" for document in documents
        )
        return (
            "Use the following context to answer the question.\n\n"
            f"<context>\n{context}\n</context>\n\n"
            f"Question: {question}"
        )

    def generate(
        self, prompt: str, max_new_tokens: int | None = None
    ) -> tuple[str, list[str]]:
        documents = self.retriever.retrieve(prompt, self.top_k)
        augmented_prompt = self._augment_prompt(prompt, documents)
        response, _ = self.chatbot.generate(
            prompt=augmented_prompt,
            max_new_tokens=max_new_tokens,
        )
        sources = [
            document.source.strip()
            for document in documents
            if document.source is not None and document.source.strip()
        ]
        return response, sources
