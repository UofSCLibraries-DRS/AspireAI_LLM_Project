import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.api.chatbots.base import Chatbot, ChatbotSource
from src.api.chatbots.bedrock import BedrockChatbot, BedrockModelNotReadyError
from src.api.chatbots.huggingface import HuggingFaceChatbot
from src.api.chatbots.rag import (
    KeywordRetriever,
    RAGChatbot,
    RAGSettings,
)
from src.api.chatbots.safechat import SafeChat
from src.api.chatbots.sgc import SGCChatbot
from src.api.warmup import WarmupCoordinator, WarmupStatus

# Load environment variables
load_dotenv()


# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt/question")
    model: Literal["M8", "M14", "LLAMA", "RAG", "SC", "SGC"] = Field(
        ..., description="Model to use for generation"
    )
    max_new_tokens: int | None = Field(
        None,
        description="Maximum tokens to generate (uses model default if not provided)",
    )


class GenerateResponse(BaseModel):
    model: str
    prompt: str
    text: str
    sources: list[ChatbotSource]
    max_new_tokens: int | None
    confidence: float | None = Field(default=None, ge=0, le=1)


# Initialize all chatbots at startup
chatbots: dict[str, Chatbot] = {}
warmup_coordinator: WarmupCoordinator | None = None
logger = logging.getLogger(__name__)
NO_CACHE = {"Cache-Control": "no-store"}


async def startup_event() -> None:
    """Initialize all chatbots when the API starts."""
    global warmup_coordinator
    warmup_coordinator = None
    print("Initializing chatbots...")

    # Initialize M8 (HuggingFace)
    try:
        chatbots["M8"] = HuggingFaceChatbot(
            id="M8", config_path="configs/chatbots/m8_hf.yaml"
        )
        print("M8 (HuggingFace) initialized")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to initialize M8: {e}")

    # Initialize M14 (Bedrock).
    try:
        chatbots["M14"] = BedrockChatbot(
            id="M14", config_path="configs/chatbots/m14_bedrock.yaml"
        )
        print("M14 (Bedrock) initialized")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to initialize M14: {e}")

    m14 = chatbots.get("M14")
    if isinstance(m14, BedrockChatbot):
        try:
            warmup_coordinator = WarmupCoordinator(m14)
            m14.on_success = warmup_coordinator.mark_success
        except Exception:
            logger.exception("Unable to initialize Bedrock warm-up coordination")

    try:
        chatbots["LLAMA"] = BedrockChatbot(
            id="LLAMA", config_path="configs/chatbots/llama_bedrock.yaml"
        )
        print("LLAMA (Bedrock) initialized")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to initialize LLAMA: {e}")

    # Use the fine-tuned M14 model for BM25 keywords and grounded answers.
    try:
        m14 = chatbots.get("M14")
        if not isinstance(m14, BedrockChatbot):
            raise RuntimeError("M14 must initialize before RAG")  # noqa: TRY004
        rag_settings = RAGSettings.from_env()
        retriever = KeywordRetriever(
            database_url=rag_settings.database_url,
            chatbot=m14,
            search_column=rag_settings.search_column,
            index_name=rag_settings.index_name,
            keyword_count=rag_settings.keyword_count,
            max_new_tokens=rag_settings.keyword_max_new_tokens,
        )
        chatbots["RAG"] = RAGChatbot(
            id="RAG",
            chatbot=m14,
            retriever=retriever,
            top_k=rag_settings.top_k,
            max_context_chars=rag_settings.max_context_chars,
        )
        print(
            f"RAG initialized (model=M14, retriever=keyword-bm25, "
            f"search_column={rag_settings.search_column}, "
            f"top_k={rag_settings.top_k})"
        )
    except Exception as e:  # noqa: BLE001
        print(f"Failed to initialize RAG: {e}")

    # Initialize SafeChat
    try:
        chatbots["SC"] = SafeChat(id="SC", config_path="configs/chatbots/safechat.yaml")
        print("SafeChat initialized")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to initialize SafeChat: {e}")

    # Initialize SafeGenChat (a separate local HTTP service).
    try:
        chatbots["SGC"] = SGCChatbot(id="SGC", config_path="configs/chatbots/sgc.yaml")
        print("SGC initialized")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to initialize SGC: {e}")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application-wide startup and shutdown behavior."""
    await startup_event()
    yield


app = FastAPI(title="AspireAI Chatbot API", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root():
    """Returns API status and available models"""
    return {
        "available_models": list(chatbots.keys()),
    }


@app.get("/models")
async def list_models():
    """List all available models."""
    return {
        "models": [
            {"id": bot_id, "type": type(bot).__name__}
            for bot_id, bot in chatbots.items()
        ]
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest = Body(...)):  # noqa: B008
    """
    Generate a response from the specified model.

    Args:
        req: GenerateRequest with prompt, model name, and optional max_new_tokens

    Returns:
        GenerateResponse with the model's output
    """
    # Check if model exists
    if req.model not in chatbots:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{req.model}' not found. Available models: {list(chatbots.keys())}",
        )

    # Get the chatbot
    chatbot = chatbots[req.model]

    if req.model in {"M14", "RAG", "SGC"}:
        state = await run_in_threadpool(warmup_state, start=True)
        if state.status != "ready":
            raise model_unavailable(state)

    try:
        # Generate response
        confidence = None
        if req.model == "M14" and isinstance(chatbot, BedrockChatbot):
            response, confidence = await run_in_threadpool(
                chatbot.generate_with_confidence,
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
            )
            sources = []
        else:
            response, sources = await run_in_threadpool(
                chatbot.generate, prompt=req.prompt, max_new_tokens=req.max_new_tokens
            )

        return GenerateResponse(
            model=req.model,
            prompt=req.prompt,
            text=response,
            sources=sources,
            max_new_tokens=req.max_new_tokens,
            confidence=confidence,
        )

    except BedrockModelNotReadyError:
        if req.model in {"M14", "RAG", "SGC"}:
            state = await run_in_threadpool(warmup_state, start=True, invalidate=True)
            raise model_unavailable(state)
        raise HTTPException(status_code=503, detail={"code": "model_unavailable"})
    except Exception as e:  # noqa: BLE001
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error generating response: {e!s}")


def get_warmup_coordinator() -> WarmupCoordinator:
    if warmup_coordinator is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "The model is unavailable.",
            },
            headers=NO_CACHE,
        )
    return warmup_coordinator


def model_unavailable(state: WarmupStatus) -> HTTPException:
    warming = state.status != "failed"
    return HTTPException(
        status_code=503,
        detail={
            "code": "model_warming" if warming else "model_unavailable",
            "message": "Model is starting…" if warming else state.message,
        },
        headers={**NO_CACHE, "Retry-After": "2" if warming else "30"},
    )


def warmup_state(*, start: bool, invalidate: bool = False) -> WarmupStatus:
    coordinator = get_warmup_coordinator()
    try:
        if invalidate:
            coordinator.invalidate()
        return coordinator.start() if start else coordinator.status()
    except (OSError, RuntimeError):
        logger.exception("Bedrock warm-up coordinator unavailable")
        raise model_unavailable(
            WarmupStatus(status="failed", message="The model is unavailable.")
        )


def warmup_response(*, start: bool) -> JSONResponse:
    try:
        state = warmup_state(start=start)
    except HTTPException:
        state = WarmupStatus(status="failed", message="The model is unavailable.")
        return JSONResponse(
            status_code=503, content=state.model_dump(), headers=NO_CACHE
        )
    code = 200
    if start:
        code = (
            202
            if state.status == "warming"
            else 503
            if state.status == "failed"
            else 200
        )
    return JSONResponse(status_code=code, content=state.model_dump(), headers=NO_CACHE)


@app.post("/warmup", response_model=WarmupStatus)
def start_warmup():
    """Start or join M14 restoration without holding the HTTP request open."""
    return warmup_response(start=True)


@app.get("/warmup", response_model=WarmupStatus)
def read_warmup():
    """Read readiness only; polling never invokes Bedrock or refreshes readiness."""
    return warmup_response(start=False)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
