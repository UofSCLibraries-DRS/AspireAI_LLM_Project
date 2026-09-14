import unittest
from pathlib import Path
from types import SimpleNamespace

from fine_tuning.inference.batched_inference import (
    _decode_continuation,
    _generation_eos_token_ids,
    _handle_prompt_formatting,
)


CONFIG_DIR = Path(__file__).parents[1] / "config"


class FakeTokenizer:
    eos_token_id = 128009
    all_special_tokens = ["<|eot_id|>", "<|end_of_text|>"]

    def get_vocab(self):
        return {"<|eot_id|>": 128009}

    def convert_tokens_to_ids(self, token):
        return self.get_vocab()[token]

    def decode(self, token_ids, **kwargs):
        self.decode_kwargs = kwargs
        return "answer<|eot_id|>ignored"


class LlamaInferenceTest(unittest.TestCase):
    def test_m13_prompt_formats_use_the_llama_instruct_wire_format(self):
        expected_prefix = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        )
        expected_suffix = (
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "Question?<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        for filename in ("base.yaml", "short_response.yaml"):
            with self.subTest(filename=filename):
                prompt, stops = _handle_prompt_formatting(
                    "Question?",
                    str(CONFIG_DIR / "prompts/llama-it" / filename),
                )
                self.assertTrue(prompt.startswith(expected_prefix))
                self.assertTrue(prompt.endswith(expected_suffix))
                self.assertIn("<|eot_id|>", stops)

    def test_generation_uses_model_terminators_and_llama_eot(self):
        tokenizer = FakeTokenizer()
        model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[128001, 128008]),
            config=SimpleNamespace(eos_token_id=None),
        )

        self.assertEqual(
            _generation_eos_token_ids(model, tokenizer),
            [128001, 128008, 128009],
        )

    def test_decodes_only_the_passed_continuation_and_honors_stops(self):
        tokenizer = FakeTokenizer()

        decoded = _decode_continuation(
            tokenizer,
            [20, 128009, 21],
            ["<|eot_id|>"],
        )

        self.assertEqual(decoded, "answer")
        self.assertFalse(tokenizer.decode_kwargs["skip_special_tokens"])
        self.assertFalse(tokenizer.decode_kwargs["clean_up_tokenization_spaces"])


if __name__ == "__main__":
    unittest.main()
