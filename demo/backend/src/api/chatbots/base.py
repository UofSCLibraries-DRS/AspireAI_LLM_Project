from abc import ABC, abstractmethod
from typing import TypeAlias, TypedDict


class TextSource(TypedDict):
    """Source content that can be displayed without a public URL."""

    title: str
    description: str | None
    transcript: str


ChatbotSource: TypeAlias = str | TextSource


class Chatbot(ABC):
    def __init__(
        self,
        id: str,
    ):
        self.id = id

    @abstractmethod
    def generate(
        self, prompt: str, max_new_tokens: int | None
    ) -> tuple[str, list[ChatbotSource]]:
        """
        Return the response and URL or raw-text sources used to produce it.
        """
