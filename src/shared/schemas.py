from datetime import datetime, timezone

from pydantic import BaseModel, Field
from typing import List, Optional

from src.shared.enums import InteractionType


class HealthResponse(BaseModel):
    status: str
    db_connection: str
    sheets_connection: str


class InteractionMessage(BaseModel):
    role: InteractionType
    message: str
    tool_calls: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class InteractionRequest(BaseModel):
    sessionId: str = Field(..., min_length=4)
    message: InteractionMessage


class InteractionResponse(BaseModel):
    sessionId: str
    messages: List[InteractionMessage]
    toolCall: Optional[str] = None
    states: List[str] = None
