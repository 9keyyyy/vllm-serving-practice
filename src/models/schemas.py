from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    response: str
    model: str
    usage: Dict[str, int]
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BatchChatRequest(BaseModel):
    """배치 처리를 위한 여러 채팅 요청"""

    requests: List[ChatRequest]
    batch_id: Optional[str] = None


class BatchChatResponse(BaseModel):
    batch_id: str
    responses: List[ChatResponse]
    total_latency_ms: float
    batch_size: int
    throughput: float  # requests per second


class HealthResponse(BaseModel):
    status: str
    vllm_connected: bool
    model: str
