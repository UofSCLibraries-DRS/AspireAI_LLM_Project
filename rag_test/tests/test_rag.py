import tempfile
import unittest
from pathlib import Path

from utils.config import RagConfig
from utils.rag import build_non_rag_prompts


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

    def test_uses_prompt_template_with_empty_retrieved_context(self) -> None:
        prompts = build_non_rag_prompts(
            eval_rows=[
                {"question": "What happened?"},
                {"question": "Why?"},
            ],
            rag_config=RagConfig(
                embedding_model="embedding-model",
                top_k=3,
                prompt_template_path=str(self.prompt_template_path),
            ),
            question_column="question",
        )

        self.assertEqual(
            prompts,
            [
                "Use the provided texts to answer the question.\n"
                "Texts:\n"
                "\n"
                "Question: What happened?\n",
                "Use the provided texts to answer the question.\n"
                "Texts:\n"
                "\n"
                "Question: Why?\n",
            ],
        )


if __name__ == "__main__":
    unittest.main()
