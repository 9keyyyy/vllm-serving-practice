from fastapi import FastAPI, Request

from src.services.batch_handler import BatchHandler
from src.services.vllm_client import VLLMClient


def get_vllm_client(request: Request) -> VLLMClient:
    """vLLM 클라이언트 의존성"""
    app: FastAPI = request.app
    if not hasattr(app.state, "vllm_client"):
        raise RuntimeError("vLLM client not initialized")
    return app.state.vllm_client


def get_batch_handler(request: Request) -> BatchHandler:
    """배치 핸들러 의존성"""
    app: FastAPI = request.app
    if not hasattr(app.state, "batch_handler"):
        raise RuntimeError("Batch handler not initialized")
    return app.state.batch_handler
