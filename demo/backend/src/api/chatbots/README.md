### Bedrock Class(es)

Bedrock has two different APIs for calling models: `InvokeModel` and `Converse`.

`BedrockChatbot` uses `InvokeModel` with an explicit prompt template for each
model. M14 is fine-tuned from Llama 3.1 8B Instruct and uses
`configs/prompt_templates/m14.yaml`; M9's original base-model template remains
in `configs/prompt_templates/m9.yaml`. Both work with the same wrapper.
Configured stop sequences trim generated text locally in `_post_process`.


### Safechat Class

Safechat is currently running on a seperate process. Making it run in the same process as the API would make things significantly easier.

### HuggingFace Class

The huggingface class should only be used on **very** small models such as `Gemma-3-270m` (the base for M8)
