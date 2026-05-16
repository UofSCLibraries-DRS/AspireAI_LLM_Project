from abc import ABC, abstractmethod
from typing import Optional


class Chatbot(ABC):
    def __init__(
        self,
        id: str,
    ):
        self.id = id

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
    ) -> tuple[str, list[str]]:
        """
        Returns response and list of sources.
        """
        pass

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: Optional[int],
    ) -> list[tuple[str, list[str]]]:
        """
        Returns one response/source tuple per prompt.
        """
        return [
            self.generate(prompt=prompt, max_new_tokens=max_new_tokens)
            for prompt in prompts
        ]
