from collections.abc import Sequence
from typing import Any


LLAMA_EOT_TOKEN = "<|eot_id|>"
LLAMA_FINETUNE_PAD_TOKEN = "<|finetune_right_pad_id|>"
LLAMA_HEADER_TOKEN = "<|start_header_id|>"


def known_token_id(tokenizer: Any, token: str) -> int | None:
    """Return a token ID only when ``token`` is present in the tokenizer."""
    vocabulary = tokenizer.get_vocab()
    if token not in vocabulary:
        return None
    return tokenizer.convert_tokens_to_ids(token)


def configure_padding_token(tokenizer: Any) -> None:
    """Prefer Llama's dedicated fine-tuning pad token over an EOS fallback."""
    if tokenizer.pad_token_id is not None:
        return

    if known_token_id(tokenizer, LLAMA_FINETUNE_PAD_TOKEN) is not None:
        tokenizer.pad_token = LLAMA_FINETUNE_PAD_TOKEN
        return

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has neither a padding token nor an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_causal_lm_batch(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
) -> dict[str, list[list[int]]]:
    """Tokenize fixed-length causal-LM examples without masking real EOS/EOT tokens."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a padding token before tokenization.")

    normalized_texts = [
        text.rstrip() if LLAMA_HEADER_TOKEN in text else text for text in texts
    ]
    for text in normalized_texts:
        if LLAMA_HEADER_TOKEN in text and not text.endswith(LLAMA_EOT_TOKEN):
            raise ValueError(
                "Llama chat-formatted training rows must end with <|eot_id|>."
            )

    encoded = tokenizer(
        normalized_texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    eot_token_id = known_token_id(tokenizer, LLAMA_EOT_TOKEN)

    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for text, encoded_ids in zip(normalized_texts, encoded["input_ids"], strict=True):
        input_ids = list(encoded_ids)
        ends_with_eot = text.endswith(LLAMA_EOT_TOKEN)

        if ends_with_eot and (
            eot_token_id is None or not input_ids or input_ids[-1] != eot_token_id
        ):
            raise ValueError(
                "The tokenizer did not encode the terminal <|eot_id|> as its special token."
            )

        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is not None and (not input_ids or input_ids[0] != bos_token_id):
            input_ids.insert(0, bos_token_id)

        was_truncated = len(input_ids) > max_length
        input_ids = input_ids[:max_length]
        if was_truncated and ends_with_eot:
            input_ids[-1] = eot_token_id

        content_length = len(input_ids)
        padding_length = max_length - content_length
        attention_mask = [1] * content_length + [0] * padding_length
        labels = input_ids.copy() + [-100] * padding_length
        input_ids += [tokenizer.pad_token_id] * padding_length

        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)

    return batch
