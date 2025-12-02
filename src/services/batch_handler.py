import asyncio
import logging
import time
from collections import deque
from typing import Dict

from src.api.middleware.metrics import record_batch_metrics
from src.config import settings
from src.models.schemas import ChatRequest, ChatResponse
from src.services.vllm_client import VLLMClient

logger = logging.getLogger(__name__)


class BatchHandler:
    """
    요청을 모아서 배치로 처리하는 핸들러

    vLLM의 continuous batching을 최대한 활용하기 위해,
    vLLM Engine의 max-num-seqs 값과 batch_max_size 값을 동일하게 설정 필요
    """

    def __init__(self, vllm_client: VLLMClient):
        self.client = vllm_client
        self.queue: deque = deque()
        self.max_batch_size = settings.batch_max_size
        self.timeout_ms = settings.batch_timeout_ms
        self._processing = False
        self._stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
        }

    async def add_request(self, request: ChatRequest) -> ChatResponse:
        """
        요청을 큐에 추가하고 배치 처리 결과 대기
        """
        future = asyncio.Future()
        self.queue.append((request, future))
        self._stats["total_requests"] += 1

        # 배치 처리 시작 (없으면)
        if not self._processing:
            asyncio.create_task(self._process_batch())

        # 결과 대기
        return await future

    async def _process_batch(self):
        """배치 처리 로직"""
        if self._processing:
            return

        self._processing = True

        try:
            # 타임아웃 또는 최대 배치 크기까지 대기
            await asyncio.sleep(self.timeout_ms / 1000)

            if not self.queue:
                return

            # 큐에서 배치 추출
            batch = []
            futures = []

            while self.queue and len(batch) < self.max_batch_size:
                request, future = self.queue.popleft()
                batch.append(request)
                futures.append(future)

            if not batch:
                return

            batch_size = len(batch)
            logger.info(f"Processing batch of {batch_size} requests")

            # vLLM으로 배치 전송
            start_time = time.perf_counter()
            responses = await self.client.batch_chat_completion(batch)
            total_time = (time.perf_counter() - start_time) * 1000

            # 📊 메트릭 기록
            record_batch_metrics(
                batch_size=batch_size, duration_seconds=total_time / 1000
            )

            # 통계 업데이트
            self._stats["total_batches"] += 1
            self._stats["avg_batch_size"] = (
                self._stats["avg_batch_size"] * (self._stats["total_batches"] - 1)
                + batch_size
            ) / self._stats["total_batches"]

            logger.info(
                f"Batch completed: {batch_size} requests in {total_time:.2f}ms "
                f"({batch_size / (total_time / 1000):.2f} req/s)"
            )

            # Future에 결과 전달
            for future, response in zip(futures, responses):
                if not future.done():
                    future.set_result(response)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # 에러 시 모든 future에 에러 전달
            for _, future in list(self.queue):
                if not future.done():
                    future.set_exception(e)

        finally:
            self._processing = False

            # 큐에 남은 요청이 있으면 다시 처리
            if self.queue:
                asyncio.create_task(self._process_batch())

    def get_stats(self) -> Dict:
        """배치 처리 통계"""
        return self._stats.copy()
