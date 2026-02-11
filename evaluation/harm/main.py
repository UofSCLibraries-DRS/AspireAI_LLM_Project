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

MAX_CTX = 2048

# question,answer_short,answer_ideal,answer_short_agg,answer_ideal_agg,dataset,subset


def main():
    criterion = CRITERIA_IDS[0]

    llm = Llama(
        model_path="/home/john/Research/ai4s/granite-guardian/model_gguf/granite-guardian-3.3-8b-Q8_0.gguf",
        n_ctx=MAX_CTX,
        logits_all=True,
        verbose=False,
    )

    # Get template from GGUF metadata
    template_str = llm.metadata["tokenizer.chat_template"]
    template = Template(template_str)

    messages = [
        {"role": "user", "content": "Summarize the text."},
        {"role": "assistant", "content": "No"},
    ]

    prompt = template.render(
        messages=messages,
        guardian_config={"criteria_id": criterion},
        add_generation_prompt=True,
        think=False,
        available_tools=None,
        documents=None,
        controls=None,
    )

    output = llm(
        prompt,
        temperature=0.0,
        max_tokens=32,
    )

    print(output["choices"][0]["text"])


if __name__ == "__main__":
    main()
