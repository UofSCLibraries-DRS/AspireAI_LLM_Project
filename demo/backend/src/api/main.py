from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.api.chatbots.bedrock import BedrockChatbot
from src.api.chatbots.huggingface import HuggingFaceChatbot
from src.api.chatbots.rag import (
    E5QueryEmbedder,
    PostgresContextRetriever,
    RAGChatbot,
    RAGSettings,
)
from src.api.chatbots.safechat import SafeChat

# Load environment variables
load_dotenv()

app = FastAPI(title="AspireAI Chatbot API", version="1.0.0")


# Request/Response Models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt/question")
    model: Literal["M8", "M9", "LLAMA", "RAG", "SC"] = Field(
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
    sources: list[str]
    max_new_tokens: int | None


# Initialize all chatbots at startup
chatbots = {}


@app.on_event("startup")
async def startup_event():
    """Initialize all chatbots when the API starts."""
    print("Initializing chatbots...")

    # Initialize M8 (HuggingFace)
    try:
        chatbots["M8"] = HuggingFaceChatbot(
            id="M8", config_path="configs/chatbots/m8_hf.yaml"
        )
        print("M8 (HuggingFace) initialized")
    except Exception as e:
        print(f"Failed to initialize M8: {e}")

    # Initialize M9 (Bedrock)
    try:
        chatbots["M9"] = BedrockChatbot(
            id="M9", config_path="configs/chatbots/m9_bedrock.yaml"
        )
        chatbots["LLAMA"] = BedrockChatbot(
            id="LLAMA", config_path="configs/chatbots/llama_bedrock.yaml"
        )
        print("M9 (Bedrock) initialized")
    except Exception as e:
        print(f"Failed to initialize M9: {e}")

    # Initialize RAG with the existing Bedrock Llama generator.
    try:
        llama = chatbots.get("LLAMA")
        if not isinstance(llama, BedrockChatbot):
            raise RuntimeError("LLAMA must initialize before RAG")
        rag_settings = RAGSettings.from_env()
        query_embedder = E5QueryEmbedder(device=rag_settings.embedding_device)
        retriever = PostgresContextRetriever(
            database_url=rag_settings.database_url,
            embedder=query_embedder,
            search_field=rag_settings.search_field,
        )
        chatbots["RAG"] = RAGChatbot(
            id="RAG",
            chatbot=llama,
            retriever=retriever,
            top_k=rag_settings.top_k,
            max_context_chars=rag_settings.max_context_chars,
        )
        print(
            f"RAG initialized (search_field={rag_settings.search_field}, "
            f"top_k={rag_settings.top_k})"
        )
    except Exception as e:
        print(f"Failed to initialize RAG: {e}")

    # Initialize SafeChat
    try:
        chatbots["SC"] = SafeChat(id="SC", config_path="configs/chatbots/safechat.yaml")
        print("SafeChat initialized")
    except Exception as e:
        print(f"Failed to initialize SafeChat: {e}")


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
async def generate(req: GenerateRequest = Body(...)):
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

    try:
        # Generate response
        response, sources = chatbot.generate(
            prompt=req.prompt, max_new_tokens=req.max_new_tokens
        )

        return GenerateResponse(
            model=req.model,
            prompt=req.prompt,
            text=response,
            sources=sources,
            max_new_tokens=req.max_new_tokens,
        )

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error generating response: {e!s}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
