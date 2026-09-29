import tempfile
import unittest
from pathlib import Path

from utils.config import RagConfig, RetrieverSpec
from utils.rag import build_non_rag_prompts, build_rag_prompts


class BuildNonRagPromptsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.prompt_template_path = Path(self.temp_dir.name) / "prompt.yaml"
        self.prompt_template_path.write_text(
            "template: |\n"
            "  Use the provided texts to answer the question.\n"
            "  Texts:\n"
            "  {retrieved_context}\n"
            "  Question: {query}\n",
            encoding="utf-8",
        )

    def test_uses_unmodified_questions(self) -> None:
        prompts = build_non_rag_prompts(
            eval_rows=[
                {"question": "What happened?"},
                {"question": "Why?"},
            ],
            question_column="question",
        )

        self.assertEqual(prompts, ["What happened?", "Why?"])

    def test_retrieves_with_the_question_text(self) -> None:
        retriever = RecordingRetriever()

        prompts = build_rag_prompts(
            eval_rows=[{"question": "What happened?"}],
            retriever=retriever,
            rag_config=RagConfig(
                top_k=2,
                retrievers=[RetrieverSpec("unused", "Unused", {})],
                prompt_template_path=str(self.prompt_template_path),
                retrieved_item_template_path=str(self._item_template()),
            ),
            question_column="question",
        )

        self.assertEqual(retriever.calls, [("What happened?", 2)])
        self.assertIn("Retrieved text", prompts[0])

    def _item_template(self) -> Path:
        path = Path(self.temp_dir.name) / "item.yaml"
        path.write_text("template: '{text}'\n", encoding="utf-8")
        return path


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[dict[str, str]]:
        self.calls.append((query, top_k))
        return [{"text": "Retrieved text"}]


if __name__ == "__main__":
    unittest.main()
