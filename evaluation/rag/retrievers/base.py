from typing import Protocol

from chatbots.base import Chatbot


class Retriever(Protocol):
    chatbot: Chatbot

    def retrieve(self, query: str, top_k: int) -> list[dict[str, str]]:
        """Return prompt-ready records ordered by relevance."""
        ...
