from chatbots.base import Chatbot
from retrievers.agentic import (
    HyDEAnswerRetriever,
    HyDEDocumentRetriever,
    HybridRetriever,
    KeywordRetriever,
)
from retrievers.base import Retriever
from retrievers.naive import NaiveRetriever
from utils.config import RetrieverSpec


RETRIEVER_BACKENDS = {
    "HyDEAnswerRetriever": HyDEAnswerRetriever,
    "HyDEDocumentRetriever": HyDEDocumentRetriever,
    "HybridRetriever": HybridRetriever,
    "KeywordRetriever": KeywordRetriever,
    "NaiveRetriever": NaiveRetriever,
}


def create_retriever(spec: RetrieverSpec, chatbot: Chatbot) -> Retriever:
    try:
        retriever_cls = RETRIEVER_BACKENDS[spec.backend]
    except KeyError as exc:
        raise ValueError(f"Unknown retriever backend `{spec.backend}`") from exc

    return retriever_cls(chatbot=chatbot, **spec.config)
