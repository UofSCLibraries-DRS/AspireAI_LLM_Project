from typing import Optional, Dict
import re

from llama_cpp import Llama
from jinja2 import Template

CRITERIA_IDS = [
    "harm",
    "social_bias",
    "profanity",
    "sexual_content",
    "unethical_behavior",
    "violence",
]


def make_prompt_renderer(llm):
    template = Template(llm.metadata["tokenizer.chat_template"])

    def render(messages, guardian_config, think=False):
        return template.render(
            messages=messages,
            guardian_config=guardian_config,
            add_generation_prompt=True,
            think=think,
            available_tools=None,
            documents=None,
            controls=None,
        )

    return render


def parse_yes_no(text: str) -> bool:
    if not text:
        return False

    # Normalize
    text = text.strip().lower()

    # Extract first word (safer than substring search)
    match = re.search(r"\b(yes|no)\b", text)

    if not match:
        # If model deviates from spec, you can decide default behavior
        raise ValueError(f"Unexpected guardian output: {text}")

    return match.group(1) == "yes"


class HarmWrapper:
    def __init__(
        self,
        model_path: str,
        max_ctx: int,
        temperature: float = 0.0,
        max_tokens: int = 32,
    ):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=max_ctx,
            logits_all=True,
            verbose=False,
        )

        # Get template from GGUF metadata
        self.render_prompt = make_prompt_renderer(self.llm)

    def evaluate_harm(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
    ) -> Dict[str, bool]:
        messages = []

        if query is not None:
            messages.append({"role": "user", "content": query})

        if response is not None:
            messages.append({"role": "assistant", "content": response})

        if not messages:
            return {}

        result = {}

        for criterion in CRITERIA_IDS:
            prompt = self.render_prompt(
                messages=messages,
                guardian_config={"criteria_id": criterion},
            )

            output = self.llm(
                prompt,
                temperature=0.0,
                max_tokens=32,
            )["choices"][0]["text"]

            result[criterion] = parse_yes_no(output)

        return result
