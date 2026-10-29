"""Bound and format conversation history shared by the generation routes."""

from typing import Literal

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1)


def recent_context(context: list[ConversationTurn] | None) -> list[ConversationTurn]:
    """Keep the latest three completed user/assistant interactions."""
    pairs: list[tuple[ConversationTurn, ConversationTurn]] = []
    question = None
    for turn in context or []:
        if turn.role == "user":
            question = turn
        elif question is not None:
            pairs.append((question, turn))
            question = None
    return [turn for pair in pairs[-3:] for turn in pair]


def format_context(context: list[ConversationTurn]) -> str:
    if not context:
        return ""
    return "Conversation history:\n" + "\n".join(
        f"{turn.role}: {turn.text}" for turn in context
    )


def conversation_prompt(prompt: str, context: list[ConversationTurn]) -> str:
    history = format_context(context)
    return f"{history}\n\nQuestion:\n{prompt}" if history else prompt
