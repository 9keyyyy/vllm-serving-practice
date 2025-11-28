import asyncio
import logging
import time
from typing import List, Optional

from openai import AsyncOpenAI

from src.config import settings
from src.models.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class VLLMClient:
    """vLLM 서버와 통신하는 비동기 클라이언트"""

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=f"{settings.vllm_base_url}/v1",
            api_key=settings.vllm_api_key,
        )
        self.model = settings.vllm_model

    async def chat_completion(
        self, request: ChatRequest, request_id: Optional[str] = None
    ) -> ChatResponse:
        """단일 채팅 요청 처리"""
        start_time = time.perf_counter()

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[msg.model_dump() for msg in request.messages],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return ChatResponse(
                id=request_id or completion.id,
                response=completion.choices[0].message.content,
                model=completion.model,
                usage=completion.usage.model_dump(),
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            raise

    async def batch_chat_completion(
        self, requests: List[ChatRequest]
    ) -> List[ChatResponse]:
        """
        배치 요청 처리 - vLLM의 continuous batching 활용
        여러 요청을 동시에 보내면 vLLM이 자동으로 배치 처리
        """
        start_time = time.perf_counter()

        # 모든 요청을 동시에 실행 (vLLM이 내부적으로 배치 처리)
        tasks = [
            self.chat_completion(req, request_id=f"batch_{i}")
            for i, req in enumerate(requests)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 에러 처리
        valid_responses = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Batch request {i} failed: {resp}")
            else:
                valid_responses.append(resp)

        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Batch completed: {len(valid_responses)}/{len(requests)} requests "
            f"in {total_time:.2f}ms"
        )

        return valid_responses

    async def health_check(self) -> bool:
        """vLLM 서버 상태 확인"""
        try:
            response = await self.client.models.list()
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
