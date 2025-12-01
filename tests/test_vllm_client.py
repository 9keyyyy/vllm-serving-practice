import asyncio

import pytest

from src.models.schemas import ChatRequest, Message, MessageRole
from src.services.vllm_client import VLLMClient


@pytest.fixture
def vllm_client():
    return VLLMClient()


@pytest.mark.asyncio
async def test_single_request(vllm_client):
    """단일 요청 테스트"""
    request = ChatRequest(
        messages=[Message(role=MessageRole.USER, content="Say hello!")],
        max_tokens=50,
    )

    response = await vllm_client.chat_completion(request)

    assert response.response is not None
    assert len(response.response) > 0
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_batch_requests(vllm_client):
    """배치 요청 테스트"""
    requests = [
        ChatRequest(messages=[Message(role=MessageRole.USER, content=f"Count to {i}")])
        for i in range(1, 6)
    ]

    responses = await vllm_client.batch_chat_completion(requests)

    assert len(responses) == 5
    for resp in responses:
        assert resp.response is not None


@pytest.mark.asyncio
async def test_concurrent_requests(vllm_client):
    """동시 요청 처리 테스트 (continuous batching)"""
    request = ChatRequest(
        messages=[Message(role=MessageRole.USER, content="Hello")],
        max_tokens=50,
    )

    # 10개 요청 동시 전송
    tasks = [vllm_client.chat_completion(request) for _ in range(10)]

    import time

    start = time.perf_counter()
    responses = await asyncio.gather(*tasks)
    duration = time.perf_counter() - start

    assert len(responses) == 10
    print(f"10 concurrent requests completed in {duration:.2f}s")
    print(f"Throughput: {10 / duration:.2f} req/s")


@pytest.mark.asyncio
async def test_health_check(vllm_client):
    """헬스체크 테스트"""
    is_healthy = await vllm_client.health_check()
    assert is_healthy is True
