from typing import Optional
from abc import ABC, abstractmethod


class Chatbot(ABC):
    def __init__(
        self,
        id: str,
    ):
        self.id = id

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: Optional[int]) -> (str, list[str]):
        """
        Returns response and list of sources.
        """
        pass
