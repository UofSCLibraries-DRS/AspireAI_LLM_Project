import asyncio
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from src.api import main as api_main
from src.api.chatbots.base import Chatbot
from src.api.chatbots.bedrock import BedrockChatbot
from src.api.chatbots.conversation import ConversationTurn, recent_context
from src.api.chatbots.rag import RAGChatbot, RetrievedDocument
from src.api.chatbots.safechat import SafeChat
from src.api.chatbots.sgc import SGCChatbot
from src.api.warmup import WarmupStatus


def make_history():
    return [
        turn for index in range(5) for turn in (
            {"role": "user", "text": f"Question {index}"},
            {"role": "assistant", "text": f"Answer {index}"},
        )
    ]


class ConversationTests(unittest.TestCase):
    def generate(self, model, bot, **kwargs):
        coordinator = Mock()
        coordinator.start.return_value = WarmupStatus(status="ready")
        with (
            patch.dict(api_main.chatbots, {model: bot}, clear=True),
            patch.object(api_main, "warmup_coordinator", coordinator),
        ):
            return asyncio.run(api_main.generate(api_main.GenerateRequest(
                prompt="Current question", model=model, context=make_history(), **kwargs
            )))

    def test_m14_and_other_llms_receive_only_latest_three_completed_pairs(self):
        for name, spec in (("M14", BedrockChatbot), ("LLAMA", BedrockChatbot), ("M8", Chatbot)):
            with self.subTest(model=name):
                bot = Mock(spec=spec)
                if name == "M14":
                    bot.generate_with_confidence.return_value = ("Answer", 0.9)
                bot.generate.return_value = ("Answer", [])
                result = self.generate(name, bot)
                call = bot.generate_with_confidence if name == "M14" else bot.generate
                prompt = call.call_args.kwargs["prompt"]
                for turn in make_history()[-6:]:
                    self.assertIn(f"{turn['role']}: {turn['text']}", prompt)
                for turn in make_history()[:-6]:
                    self.assertNotIn(turn["text"], prompt)
                self.assertTrue(prompt.endswith("Question:\nCurrent question"))
                self.assertEqual(prompt.count("Current question"), 1)
                self.assertEqual(result.prompt, "Current question")

    def test_rag_uses_bounded_history_for_retrieval_and_answers(self):
        source = {"title": "Record", "description": None, "transcript": "Evidence"}
        for confidence in (False, True):
            with self.subTest(confidence=confidence):
                bot = Mock(spec=BedrockChatbot)
                bot.generate.return_value = ("Answer", [])
                bot.generate_with_confidence.return_value = ("Answer", 0.9)
                retriever = Mock()
                retriever.retrieve.return_value = [RetrievedDocument("Evidence", source)]
                rag = RAGChatbot("RAG", bot, retriever)
                result = self.generate(
                    "RAG", rag, return_confidence=confidence,
                    answer_context="Relevant examples:\nAn experience example",
                )
                query = retriever.retrieve.call_args.args[0]
                call = bot.generate_with_confidence if confidence else bot.generate
                answer_prompt = call.call_args.kwargs["prompt"]
                for text in (query, answer_prompt):
                    for turn in make_history()[-6:]:
                        self.assertIn(f"{turn['role']}: {turn['text']}", text)
                    for turn in make_history()[:-6]:
                        self.assertNotIn(turn["text"], text)
                    self.assertEqual(text.count("Current question"), 1)
                self.assertNotIn("An experience example", query)
                self.assertIn("An experience example", answer_prompt)
                self.assertIn("Evidence", answer_prompt)
                self.assertEqual(result.sources, [source])
                self.assertEqual(result.confidence, 0.9 if confidence else None)

    @patch("src.api.chatbots.sgc.requests.post")
    def test_sgc_receives_bounded_structured_history_with_current_question_separate(self, post):
        post.return_value.json.return_value = {
            "solution": "Answer", "system": "System1", "sources": []
        }
        bot = SGCChatbot("SGC", "configs/chatbots/sgc.yaml")
        self.generate("SGC", bot)
        self.assertEqual(post.call_args.kwargs["json"], {
            "problem": "Current question", "context": make_history()[-6:]
        })

    def test_safechat_still_receives_only_current_question(self):
        bot = Mock(spec=SafeChat)
        bot.generate.return_value = ("Answer", [])
        self.generate("SC", bot)
        self.assertEqual(bot.generate.call_args.kwargs["prompt"], "Current question")

    def test_incomplete_and_orphan_turns_are_excluded(self):
        history = [ConversationTurn(**turn) for turn in (
            {"role": "assistant", "text": "Orphan answer"},
            {"role": "user", "text": "Abandoned question"},
            {"role": "user", "text": "Completed question"},
            {"role": "assistant", "text": "Completed answer"},
            {"role": "assistant", "text": "Extra answer"},
            {"role": "user", "text": "Unanswered question"},
        )]
        self.assertEqual(recent_context(history), history[2:4])

    def test_context_schema_preserves_single_turn_requests_and_rejects_invalid_roles(self):
        self.assertEqual(api_main.GenerateRequest(prompt="Question", model="M14").context, [])
        with self.assertRaises(ValidationError):
            api_main.GenerateRequest(
                prompt="Question", model="M14",
                context=[{"role": "system", "text": "Invalid role"}],
            )


if __name__ == "__main__":
    unittest.main()
