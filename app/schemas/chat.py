from typing import List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        description="Sliding window of conversation turns (last N messages)"
    )
    initial_problem: str = Field(
        max_length=4000,
        description="The original math problem the user solved"
    )
    initial_solution: str = Field(
        max_length=16000,
        description="The full solver response for context"
    )
    max_turns: int = Field(
        default=10,
        ge=2,
        le=40,
        description="Max number of messages to keep in the sliding window"
    )


class ChatResponse(BaseModel):
    answer: str
    error: Optional[str] = None
