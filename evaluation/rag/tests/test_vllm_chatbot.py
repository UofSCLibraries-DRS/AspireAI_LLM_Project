import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatbots.factory import create_chatbot
from chatbots.vllm import (
    VLLM_WORKER_MULTIPROC_METHOD,
    VLLMChatbot,
    VLLMChatbotConfig,
)
from utils.config import ChatbotSpec


class VLLMChatbotConfigTests(unittest.TestCase):
    def test_loads_aliases_and_llm_kwargs(self) -> None:
        config = VLLMChatbotConfig.from_config(
            {
                "model_id": "local-model",
                "chat_template_path": "template.yaml",
                "temperature": 0.7,
                "max_tokens": 64,
                "top_p": 0.9,
                "llm_kwargs": {"tensor_parallel_size": 2},
            }
        )

        self.assertEqual(config.model_path, "local-model")
        self.assertEqual(config.prompt_template_path, "template.yaml")
        self.assertEqual(config.model_temperature, 0.7)
        self.assertEqual(config.max_tokens, 64)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.llm_kwargs, {"tensor_parallel_size": 2})
        self.assertIsNone(config.adapter_path)
        self.assertFalse(config.is_instruct)

    def test_loads_adapter_path(self) -> None:
        config = VLLMChatbotConfig.from_config(
            {
                "model_path": "local-model",
                "adapter_path": "M12",
            }
        )

        self.assertEqual(config.adapter_path, "M12")

    def test_loads_is_instruct(self) -> None:
        config = VLLMChatbotConfig.from_config(
            {
                "model_path": "local-model",
                "is_instruct": True,
            }
        )

        self.assertTrue(config.is_instruct)

    def test_rejects_non_boolean_is_instruct(self) -> None:
        with self.assertRaisesRegex(ValueError, "is_instruct"):
            VLLMChatbotConfig.from_config(
                {
                    "model_path": "local-model",
                    "is_instruct": "true",
                }
            )

    def test_prompt_template_path_defaults_to_identity(self) -> None:
        config = VLLMChatbotConfig.from_config(
            {
                "model_path": "local-model",
            }
        )

        self.assertIsNone(config.prompt_template_path)

    def test_rejects_model_in_llm_kwargs(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_path"):
            VLLMChatbotConfig.from_config(
                {
                    "model_path": "local-model",
                    "prompt_template_path": "template.yaml",
                    "llm_kwargs": {"model": "other-model"},
                }
            )


class VLLMChatbotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.template_path = Path(self.temp_dir.name) / "template.yaml"
        self.template_path.write_text(
            "template: |-\n"
            "  Question: {user_prompt}\n"
            "  Answer:\n"
            "stop_sequences: ['</answer>']\n",
            encoding="utf-8",
        )
        self.adapters_root = Path(self.temp_dir.name) / "adapters"
        self.adapter_path = self.adapters_root / "M12"
        self.adapter_path.mkdir(parents=True)

    def test_sets_spawn_before_loading_vllm(self) -> None:
        def load_vllm_classes() -> tuple[type, type, type]:
            self.assertEqual(
                os.environ.get(VLLM_WORKER_MULTIPROC_METHOD),
                "spawn",
            )
            return FakeLLM, FakeSamplingParams, FakeLoRARequest

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "chatbots.vllm._load_vllm_classes",
                side_effect=load_vllm_classes,
            ),
        ):
            VLLMChatbot(id="vllm", config={"model_path": "local-model"})

    def test_generate_batch_uses_single_vllm_batch_call(self) -> None:
        with patch(
            "chatbots.vllm._load_vllm_classes",
            return_value=(FakeLLM, FakeSamplingParams, FakeLoRARequest),
        ):
            chatbot = VLLMChatbot(
                id="vllm",
                config={
                    "model_path": "local-model",
                    "prompt_template_path": str(self.template_path),
                    "model_temperature": 0.7,
                    "max_tokens": 32,
                    "top_p": 0.9,
                    "llm_kwargs": {"tensor_parallel_size": 2},
                },
            )

        generations = chatbot.generate_batch(
            prompts=["First?", "Second?"],
            max_new_tokens=16,
        )

        self.assertEqual(chatbot.llm.model, "local-model")
        self.assertEqual(chatbot.llm.kwargs, {"tensor_parallel_size": 2})
        self.assertEqual(chatbot.llm.chat_calls, [])
        self.assertEqual(
            chatbot.llm.calls,
            [
                {
                    "prompts": [
                        "Question: First?\nAnswer:",
                        "Question: Second?\nAnswer:",
                    ],
                    "sampling_params": FakeSamplingParams(
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=16,
                        stop=["</answer>"],
                    ),
                    "use_tqdm": False,
                    "lora_request": None,
                }
            ],
        )
        self.assertEqual(
            generations,
            [
                ("Generated 1: Question: First?\nAnswer:", []),
                ("Generated 2: Question: Second?\nAnswer:", []),
            ],
        )

    def test_generate_delegates_to_generate_batch(self) -> None:
        with patch(
            "chatbots.vllm._load_vllm_classes",
            return_value=(FakeLLM, FakeSamplingParams, FakeLoRARequest),
        ):
            chatbot = VLLMChatbot(
                id="vllm",
                config={
                    "model_path": "local-model",
                    "prompt_template_path": str(self.template_path),
                    "max_tokens": 32,
                },
            )

        generation = chatbot.generate(prompt="Solo?", max_new_tokens=7)

        self.assertEqual(generation, ("Generated 1: Question: Solo?\nAnswer:", []))
        self.assertEqual(chatbot.llm.calls[0]["sampling_params"].max_tokens, 7)

    def test_generate_batch_uses_identity_prompt_when_template_is_omitted(self) -> None:
        with patch(
            "chatbots.vllm._load_vllm_classes",
            return_value=(FakeLLM, FakeSamplingParams, FakeLoRARequest),
        ):
            chatbot = VLLMChatbot(
                id="vllm",
                config={
                    "model_path": "local-model",
                },
            )

        generations = chatbot.generate_batch(
            prompts=["Already formatted prompt"],
            max_new_tokens=None,
        )

        self.assertEqual(chatbot.llm.calls[0]["prompts"], ["Already formatted prompt"])
        self.assertEqual(chatbot.llm.calls[0]["sampling_params"].stop, None)
        self.assertEqual(generations, [("Generated 1: Already formatted prompt", [])])

    def test_generate_batch_uses_chat_for_instruct_models(self) -> None:
        with patch(
            "chatbots.vllm._load_vllm_classes",
            return_value=(FakeLLM, FakeSamplingParams, FakeLoRARequest),
        ):
            chatbot = VLLMChatbot(
                id="vllm",
                config={
                    "model_path": "local-model",
                    "is_instruct": True,
                    "prompt_template_path": str(self.template_path),
                    "model_temperature": 0.7,
                    "max_tokens": 32,
                    "top_p": 0.9,
                },
            )

        generations = chatbot.generate_batch(
            prompts=["First?", "Second?"],
            max_new_tokens=16,
        )

        self.assertEqual(chatbot.llm.calls, [])
        self.assertEqual(
            chatbot.llm.chat_calls,
            [
                {
                    "conversations": [
                        [
                            {
                                "role": "user",
                                "content": "Question: First?\nAnswer:",
                            }
                        ],
                        [
                            {
                                "role": "user",
                                "content": "Question: Second?\nAnswer:",
                            }
                        ],
                    ],
                    "sampling_params": FakeSamplingParams(
                        temperature=0.7,
                        top_p=0.9,
                        max_tokens=16,
                        stop=["</answer>"],
                    ),
                    "use_tqdm": False,
                    "lora_request": None,
                }
            ],
        )
        self.assertEqual(
            generations,
            [
                ("Chat 1: Question: First?\nAnswer:", []),
                ("Chat 2: Question: Second?\nAnswer:", []),
            ],
        )

    def test_generate_batch_uses_configured_lora_adapter(self) -> None:
        with (
            patch.dict(os.environ, {"ADAPTERS_PATH": str(self.adapters_root)}),
            patch(
                "chatbots.vllm._load_vllm_classes",
                return_value=(FakeLLM, FakeSamplingParams, FakeLoRARequest),
            ),
        ):
            chatbot = VLLMChatbot(
                id="vllm",
                config={
                    "model_path": "local-model",
                    "adapter_path": "M12",
                    "llm_kwargs": {"tensor_parallel_size": 2},
                },
            )

        chatbot.generate_batch(prompts=["Hello"], max_new_tokens=None)

        self.assertEqual(
            chatbot.llm.kwargs,
            {"tensor_parallel_size": 2, "enable_lora": True},
        )
        self.assertEqual(
            chatbot.llm.calls[0]["lora_request"],
            FakeLoRARequest(
                lora_name="vllm",
                lora_int_id=1,
                lora_path=str(self.adapter_path),
            ),
        )

    def test_adapter_path_falls_back_to_adapters_folder_env_var(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"ADAPTERS_FOLDER": str(self.adapters_root)},
                clear=True,
            ),
            patch(
                "chatbots.vllm._load_vllm_classes",
                return_value=(FakeLLM, FakeSamplingParams, FakeLoRARequest),
            ),
        ):
            chatbot = VLLMChatbot(
                id="vllm",
                config={
                    "model_path": "local-model",
                    "adapter_path": "M12",
                },
            )

        self.assertEqual(chatbot.adapter_path, self.adapter_path)

    def test_rejects_missing_adapter_path(self) -> None:
        with (
            patch.dict(os.environ, {"ADAPTERS_PATH": str(self.adapters_root)}),
            patch(
                "chatbots.vllm._load_vllm_classes",
                return_value=(FakeLLM, FakeSamplingParams, FakeLoRARequest),
            ),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "Adapter path not found"):
                VLLMChatbot(
                    id="vllm",
                    config={
                        "model_path": "local-model",
                        "adapter_path": "missing",
                    },
                )

    def test_rejects_adapter_path_without_adapters_env_var(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ADAPTERS_PATH"):
                VLLMChatbot(
                    id="vllm",
                    config={
                        "model_path": "local-model",
                        "adapter_path": "M12",
                    },
                )

    def test_rejects_wrong_output_count(self) -> None:
        with patch(
            "chatbots.vllm._load_vllm_classes",
            return_value=(ShortFakeLLM, FakeSamplingParams, FakeLoRARequest),
        ):
            chatbot = VLLMChatbot(
                id="vllm",
                config={
                    "model_path": "local-model",
                    "prompt_template_path": str(self.template_path),
                },
            )

        with self.assertRaisesRegex(RuntimeError, "returned 1 outputs for 2 prompts"):
            chatbot.generate_batch(prompts=["First?", "Second?"], max_new_tokens=None)

    def test_rejects_output_without_completions(self) -> None:
        with patch(
            "chatbots.vllm._load_vllm_classes",
            return_value=(EmptyCompletionFakeLLM, FakeSamplingParams, FakeLoRARequest),
        ):
            chatbot = VLLMChatbot(
                id="vllm",
                config={
                    "model_path": "local-model",
                    "prompt_template_path": str(self.template_path),
                },
            )

        with self.assertRaisesRegex(RuntimeError, "did not contain completions"):
            chatbot.generate_batch(prompts=["First?"], max_new_tokens=None)

    def test_factory_creates_vllm_chatbot(self) -> None:
        with patch(
            "chatbots.vllm._load_vllm_classes",
            return_value=(FakeLLM, FakeSamplingParams, FakeLoRARequest),
        ):
            chatbot = create_chatbot(
                ChatbotSpec(
                    id="vllm",
                    backend="VLLMChatbot",
                    config={
                        "model_path": "local-model",
                        "prompt_template_path": str(self.template_path),
                    },
                )
            )

        self.assertIsInstance(chatbot, VLLMChatbot)


class FakeSamplingParams:
    def __init__(
        self,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FakeSamplingParams):
            return False
        return self.__dict__ == other.__dict__


class FakeLoRARequest:
    def __init__(self, lora_name: str, lora_int_id: int, lora_path: str) -> None:
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FakeLoRARequest):
            return False
        return self.__dict__ == other.__dict__


class FakeCompletion:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [FakeCompletion(text)]


class EmptyFakeRequestOutput:
    outputs: list[FakeCompletion] = []


class FakeLLM:
    def __init__(self, model: str, **kwargs: object) -> None:
        self.model = model
        self.kwargs = kwargs
        self.calls: list[dict[str, object]] = []
        self.chat_calls: list[dict[str, object]] = []

    def generate(
        self,
        prompts: list[str],
        sampling_params: FakeSamplingParams,
        use_tqdm: bool,
        lora_request: FakeLoRARequest | None = None,
    ) -> list[FakeRequestOutput]:
        self.calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "lora_request": lora_request,
            }
        )
        return [
            FakeRequestOutput(f"Generated {index}: {prompt}")
            for index, prompt in enumerate(prompts, start=1)
        ]

    def chat(
        self,
        conversations: list[list[dict[str, str]]],
        sampling_params: FakeSamplingParams,
        use_tqdm: bool,
        lora_request: FakeLoRARequest | None = None,
    ) -> list[FakeRequestOutput]:
        self.chat_calls.append(
            {
                "conversations": conversations,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "lora_request": lora_request,
            }
        )
        return [
            FakeRequestOutput(f"Chat {index}: {conversation[0]['content']}")
            for index, conversation in enumerate(conversations, start=1)
        ]


class ShortFakeLLM(FakeLLM):
    def generate(
        self,
        prompts: list[str],
        sampling_params: FakeSamplingParams,
        use_tqdm: bool,
        lora_request: FakeLoRARequest | None = None,
    ) -> list[FakeRequestOutput]:
        return [FakeRequestOutput("Generated")]


class EmptyCompletionFakeLLM(FakeLLM):
    def generate(
        self,
        prompts: list[str],
        sampling_params: FakeSamplingParams,
        use_tqdm: bool,
        lora_request: FakeLoRARequest | None = None,
    ) -> list[EmptyFakeRequestOutput]:
        return [EmptyFakeRequestOutput()]


if __name__ == "__main__":
    unittest.main()
