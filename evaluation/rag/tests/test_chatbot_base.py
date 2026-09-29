import unittest

from chatbots.base import Chatbot


class ChatbotBatchTests(unittest.TestCase):
    def test_generate_batch_calls_generate_for_each_prompt_in_order(self) -> None:
        chatbot = RecordingChatbot()

        generations = chatbot.generate_batch(
            prompts=["Prompt one", "Prompt two"],
            max_new_tokens=12,
        )

        self.assertEqual(
            chatbot.calls,
            [("Prompt one", 12), ("Prompt two", 12)],
        )
        self.assertEqual(
            generations,
            [("Response 1", []), ("Response 2", [])],
        )


class RecordingChatbot(Chatbot):
    def __init__(self) -> None:
        super().__init__(id="recording")
        self.calls: list[tuple[str, int | None]] = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None,
    ) -> tuple[str, list[str]]:
        self.calls.append((prompt, max_new_tokens))
        return f"Response {len(self.calls)}", []


if __name__ == "__main__":
    unittest.main()
