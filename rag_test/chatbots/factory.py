from chatbots.bedrock import BedrockChatbot
from chatbots.dummy import DummyChatbot
from chatbots.huggingface import HuggingFaceChatbot
from chatbots.openai import OpenAIChatbot
from utils.config import ChatbotSpec


CHATBOT_BACKENDS = {
    "BedrockChatbot": BedrockChatbot,
    "DummyChatbot": DummyChatbot,
    "HuggingFaceChatbot": HuggingFaceChatbot,
    "OpenAIChatbot": OpenAIChatbot,
}


def create_chatbot(spec: ChatbotSpec):
    """
    Instantiates class defined in `spec.backend` with provided config
    """
    try:
        chatbot_cls = CHATBOT_BACKENDS[spec.backend]
    except KeyError as exc:
        raise ValueError(f"Unknown chatbot backend `{spec.backend}`") from exc

    return chatbot_cls(id=spec.id, config=spec.config)
