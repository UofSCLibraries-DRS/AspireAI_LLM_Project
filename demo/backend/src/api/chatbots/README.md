### Bedrock Class(es)

Bedrock has two different APIs for calling models: `InvokeModel` and `Converse`.

Since we are fine-tuning from base models, we are locked in to using the `InvokeModel` API, which, for Llama 3.1, does not support stop sequences.

If we choose to use baseline models through Bedrock, we will need to have two classes:

 - `BedrockInvokeModelChatbot`
 - `BedrockConverseChatbot`


### Safechat Class

Safechat is currently running on a seperate process. Making it run in the same process as the API would make things significantly easier.

### HuggingFace Class

The huggingface class should only be used on **very** small models such as `Gemma-3-270m` (the base for M8)