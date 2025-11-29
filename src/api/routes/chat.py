from fastapi import APIRouter, Depends, HTTPException

from src.api.main import get_batch_handler, get_vllm_client
from src.models.schemas import ChatRequest, ChatResponse
from src.services.batch_handler import BatchHandler
from src.services.vllm_client import VLLMClient

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client: VLLMClient = Depends(get_vllm_client),
):
    """단일 채팅 요청 (배치 미사용)"""
    try:
        return await client.chat_completion(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/batch", response_model=ChatResponse)
async def chat_with_batch(
    request: ChatRequest,
    batch_handler: BatchHandler = Depends(get_batch_handler),
):
    """
    배치 처리를 사용하는 채팅
    - 여러 요청이 동시에 들어오면 자동으로 배치 처리
    - vLLM의 continuous batching 최대 활용
    """
    try:
        return await batch_handler.add_request(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
