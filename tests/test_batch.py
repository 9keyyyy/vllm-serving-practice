import asyncio

import pytest

from src.models.schemas import ChatRequest, Message, MessageRole
from src.services.batch_handler import BatchHandler
from src.services.vllm_client import VLLMClient


@pytest.fixture
def batch_handler():
    client = VLLMClient()
    return BatchHandler(client)


@pytest.mark.asyncio
async def test_batch_accumulation(batch_handler):
    """배치 누적 테스트"""
    request = ChatRequest(
        messages=[Message(role=MessageRole.USER, content="Test")],
        max_tokens=50,
    )

    # 5개 요청을 거의 동시에 전송
    tasks = [batch_handler.add_request(request) for _ in range(5)]
    responses = await asyncio.gather(*tasks)

    assert len(responses) == 5

    # 통계 확인
    stats = batch_handler.get_stats()
    assert stats["total_requests"] == 5
    assert stats["total_batches"] >= 1


@pytest.mark.asyncio
async def test_batch_timeout(batch_handler):
    """배치 타임아웃 테스트"""
    request = ChatRequest(
        messages=[Message(role=MessageRole.USER, content="Test")],
        max_tokens=50,
    )

    # 단일 요청 (타임아웃 후 처리되어야 함)
    import time

    start = time.perf_counter()
    response = await batch_handler.add_request(request)
    duration = time.perf_counter() - start

    # 최소 타임아웃 시간은 지나야 함
    assert duration >= 0.1  # 100ms timeout
    assert response is not None
