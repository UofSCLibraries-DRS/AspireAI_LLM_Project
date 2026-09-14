import unittest

from fine_tuning.utils.causal_lm import (
    LLAMA_BOS_TOKEN,
    LLAMA_EOT_TOKEN,
    LLAMA_FINETUNE_PAD_TOKEN,
    configure_padding_token,
    tokenize_causal_lm_batch,
)


class FakeTokenizer:
    def __init__(self, encoded_texts, pad_token=None):
        self.encoded_texts = encoded_texts
        self.calls = []
        self.vocabulary = {
            "<|begin_of_text|>": 1,
            LLAMA_EOT_TOKEN: 2,
            LLAMA_FINETUNE_PAD_TOKEN: 3,
        }
        self.eos_token = LLAMA_EOT_TOKEN
        self.eos_token_id = 2
        self.bos_token_id = 1
        self._pad_token = pad_token
        self.pad_token_id = (
            self.vocabulary[pad_token] if pad_token is not None else None
        )

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self.pad_token_id = self.vocabulary[value]

    def get_vocab(self):
        return self.vocabulary

    def convert_tokens_to_ids(self, token):
        return self.vocabulary[token]

    def __call__(self, texts, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": [self.encoded_texts[text] for text in texts]}


class CausalLMTokenizationTest(unittest.TestCase):
    def test_configures_the_dedicated_llama_finetune_pad_token(self):
        tokenizer = FakeTokenizer({})

        configure_padding_token(tokenizer)

        self.assertEqual(tokenizer.pad_token, LLAMA_FINETUNE_PAD_TOKEN)
        self.assertNotEqual(tokenizer.pad_token_id, tokenizer.eos_token_id)

    def test_does_not_add_special_tokens_and_keeps_eot_label(self):
        text = "<|begin_of_text|>chat<|eot_id|>"
        tokenizer = FakeTokenizer({text: [1, 10, 2]}, pad_token=LLAMA_EOT_TOKEN)

        batch = tokenize_causal_lm_batch(tokenizer, [text], max_length=5)

        self.assertFalse(tokenizer.calls[0]["add_special_tokens"])
        self.assertEqual(batch["input_ids"][0], [1, 10, 2, 2, 2])
        self.assertEqual(batch["attention_mask"][0], [1, 1, 1, 0, 0])
        self.assertEqual(batch["labels"][0], [1, 10, 2, -100, -100])

    def test_preserves_terminal_eot_when_a_chat_is_truncated(self):
        text = f"{LLAMA_BOS_TOKEN}<|start_header_id|>user chat<|eot_id|>"
        tokenizer = FakeTokenizer(
            {text: [1, 4, 5, 6, 7, 8, 2]},
            pad_token=LLAMA_FINETUNE_PAD_TOKEN,
        )

        batch = tokenize_causal_lm_batch(tokenizer, [text], max_length=4)

        self.assertEqual(batch["input_ids"][0], [1, 4, 5, 2])
        self.assertEqual(batch["labels"][0], [1, 4, 5, 2])

    def test_adds_one_bos_token_to_unformatted_documents(self):
        text = "plain document"
        tokenizer = FakeTokenizer({text: [10, 11]}, pad_token=LLAMA_FINETUNE_PAD_TOKEN)

        batch = tokenize_causal_lm_batch(tokenizer, [text], max_length=4)

        self.assertEqual(batch["input_ids"][0], [1, 10, 11, 3])
        self.assertEqual(batch["labels"][0], [1, 10, 11, -100])

    def test_ignores_whitespace_after_a_chat_terminal_eot(self):
        text = f"{LLAMA_BOS_TOKEN}<|start_header_id|>assistant reply<|eot_id|>\n"
        normalized = text.rstrip()
        tokenizer = FakeTokenizer(
            {normalized: [1, 4, 5, 2]}, pad_token=LLAMA_FINETUNE_PAD_TOKEN
        )

        batch = tokenize_causal_lm_batch(tokenizer, [text], max_length=5)

        self.assertEqual(batch["input_ids"][0], [1, 4, 5, 2, 3])
        self.assertEqual(batch["labels"][0], [1, 4, 5, 2, -100])

    def test_adds_terminal_eot_when_a_llama_chat_is_missing_one(self):
        text = (
            f"{LLAMA_BOS_TOKEN}<|start_header_id|>assistant"
            "<|end_header_id|>\n\nreply"
        )
        repaired = text + LLAMA_EOT_TOKEN
        tokenizer = FakeTokenizer(
            {repaired: [1, 4, 5, 2]}, pad_token=LLAMA_FINETUNE_PAD_TOKEN
        )

        batch = tokenize_causal_lm_batch(tokenizer, [text], max_length=5)

        self.assertEqual(batch["input_ids"][0], [1, 4, 5, 2, 3])
        self.assertEqual(batch["labels"][0], [1, 4, 5, 2, -100])

    def test_does_not_treat_a_header_token_inside_raw_text_as_a_chat(self):
        text = "A transcript quotes <|start_header_id|> as literal text."
        tokenizer = FakeTokenizer({text: [10, 11]}, pad_token=LLAMA_FINETUNE_PAD_TOKEN)

        batch = tokenize_causal_lm_batch(tokenizer, [text], max_length=3)

        self.assertEqual(batch["input_ids"][0], [1, 10, 11])


if __name__ == "__main__":
    unittest.main()
