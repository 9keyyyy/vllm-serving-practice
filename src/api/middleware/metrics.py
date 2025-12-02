"""
Prometheus metrics middleware
"""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

# ============================================
# Prometheus Metrics 정의
# ============================================

# HTTP 요청
http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# 배치 처리
batch_requests_total = Counter("batch_requests_total", "Total batch requests processed")

batch_size_total = Histogram(
    "batch_size_total", "Batch size distribution", buckets=(1, 2, 4, 8, 16, 32, 64)
)

batch_duration_seconds = Histogram(
    "batch_duration_seconds",
    "Batch processing duration in seconds",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# vLLM
vllm_requests_total = Counter(
    "vllm_requests_total",
    "Total vLLM requests",
    ["status"],  # success, error
)

vllm_latency_seconds = Histogram(
    "vllm_latency_seconds",
    "vLLM request latency in seconds",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)


# ============================================
# Middleware
# ============================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI metrics middleware"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # /metrics는 제외
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        path = request.url.path
        start_time = time.time()

        # 요청 처리
        response = await call_next(request)

        # 소요 시간
        duration = time.time() - start_time

        # 메트릭 기록
        http_requests_total.labels(
            method=method, endpoint=path, status=response.status_code
        ).inc()

        http_request_duration_seconds.labels(method=method, endpoint=path).observe(
            duration
        )

        return response


# ============================================
# 헬퍼 함수
# ============================================


def record_batch_metrics(batch_size: int, duration_seconds: float):
    """배치 처리 메트릭 기록"""
    batch_requests_total.inc()
    batch_size_total.observe(batch_size)
    batch_duration_seconds.observe(duration_seconds)


def record_vllm_metrics(success: bool, latency_seconds: float):
    """vLLM 요청 메트릭 기록"""
    status = "success" if success else "error"
    vllm_requests_total.labels(status=status).inc()
    vllm_latency_seconds.observe(latency_seconds)


# ============================================
# Metrics 엔드포인트
# ============================================


def get_metrics():
    """Prometheus /metrics 엔드포인트"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
