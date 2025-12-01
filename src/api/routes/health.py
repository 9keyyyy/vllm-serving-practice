from fastapi import APIRouter, Depends

from src.api.dependencies import get_vllm_client
from src.config import settings
from src.models.schemas import HealthResponse
from src.services.vllm_client import VLLMClient

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(client: VLLMClient = Depends(get_vllm_client)):
    """서버 상태 확인"""
    vllm_connected = await client.health_check()

    return HealthResponse(
        status="healthy" if vllm_connected else "degraded",
        vllm_connected=vllm_connected,
        model=settings.vllm_model,
    )
