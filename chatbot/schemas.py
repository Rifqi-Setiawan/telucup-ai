from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    ip: str | None = None


class ChatResponse(BaseModel):
    status: str
    answer: str
    meta: dict = {}
