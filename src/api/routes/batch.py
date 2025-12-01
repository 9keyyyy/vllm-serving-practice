import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_batch_handler, get_vllm_client
from src.models.schemas import BatchChatRequest, BatchChatResponse
from src.services.batch_handler import BatchHandler
from src.services.vllm_client import VLLMClient

router = APIRouter()


@router.post("/batch/chat", response_model=BatchChatResponse)
async def batch_chat(
    request: BatchChatRequest,
    client: VLLMClient = Depends(get_vllm_client),
):
    """
    명시적 배치 처리
    - 여러 요청을 한 번에 전송
    - 대량 처리에 최적
    """
    if not request.requests:
        raise HTTPException(status_code=400, detail="Empty batch")

    batch_id = request.batch_id or str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        responses = await client.batch_chat_completion(request.requests)

        total_latency = (time.perf_counter() - start_time) * 1000
        throughput = len(responses) / (total_latency / 1000)

        return BatchChatResponse(
            batch_id=batch_id,
            responses=responses,
            total_latency_ms=total_latency,
            batch_size=len(responses),
            throughput=throughput,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch/stats")
async def batch_stats(
    batch_handler: BatchHandler = Depends(get_batch_handler),
):
    """배치 처리 통계"""
    return batch_handler.get_stats()
