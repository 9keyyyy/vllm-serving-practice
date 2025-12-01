"""
배치 처리 성능 벤치마크
vLLM의 continuous batching 효과 측정
"""

import asyncio
import random
import statistics
import time

from src.models.schemas import ChatRequest, Message, MessageRole
from src.services.batch_handler import BatchHandler
from src.services.vllm_client import VLLMClient


def generate_varied_requests(n: int):
    """다양한 길이의 요청 생성"""
    prompts = [
        ("Hi", 10),
        ("Explain AI briefly.", 30),
        ("Write a paragraph about machine learning.", 80),
        ("Write a detailed explanation of neural networks.", 150),
    ]

    requests = []
    for i in range(n):
        content, max_tokens = random.choice(prompts)
        requests.append(
            ChatRequest(
                messages=[Message(role=MessageRole.USER, content=content)],
                max_tokens=max_tokens,
            )
        )

    return requests


async def benchmark_single_requests(client: VLLMClient, requests):
    """개별 요청 벤치마크"""
    n = len(requests)
    print(f"\n=== Single Requests Benchmark (n={n}) ===")

    latencies = []
    start = time.perf_counter()

    for i in range(n):
        req_start = time.perf_counter()
        await client.chat_completion(requests[i])
        latencies.append(time.perf_counter() - req_start)
        if (i + 1) % 10 == 0:
            print(f"Request {i + 1}/{n} completed")

    total_time = time.perf_counter() - start

    print("\nResults:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {n / total_time:.2f} req/s")
    print(f"Avg latency: {statistics.mean(latencies):.2f}s")
    print(f"P50 latency: {statistics.median(latencies):.2f}s")
    print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f}s")

    return total_time, latencies


async def benchmark_concurrent_requests(client: VLLMClient, requests):
    """동시 요청 벤치마크 (vLLM continuous batching)"""
    n = len(requests)
    print(f"\n=== Concurrent Requests Benchmark (n={n}) ===")

    start = time.perf_counter()

    # 모든 요청을 동시에 전송
    tasks = [client.chat_completion(req) for req in requests]
    responses = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start
    latencies = [r.latency_ms / 1000 for r in responses]

    print("\nResults:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {n / total_time:.2f} req/s")
    print(f"Avg latency: {statistics.mean(latencies):.2f}s")
    print(f"P50 latency: {statistics.median(latencies):.2f}s")
    print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f}s")

    return total_time, latencies


async def benchmark_explicit_batch(client: VLLMClient, requests):
    """명시적 배치 벤치마크"""
    n = len(requests)
    print(f"\n=== Explicit Batch Benchmark (n={n}) ===")

    start = time.perf_counter()
    responses = await client.batch_chat_completion(requests)
    total_time = time.perf_counter() - start

    latencies = [r.latency_ms / 1000 for r in responses]

    print("\nResults:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {n / total_time:.2f} req/s")
    print(f"Avg latency: {statistics.mean(latencies):.2f}s")

    return total_time, latencies


async def benchmark_batch_handler(batch_handler: BatchHandler, requests):
    """배치 핸들러 벤치마크"""
    n = len(requests)
    print(f"\n=== Batch Handler Benchmark (n={n}) ===")

    start = time.perf_counter()

    # 요청을 빠르게 큐에 추가
    tasks = [batch_handler.add_request(req) for req in requests]
    responses = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start

    print("\nResults:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {n / total_time:.2f} req/s")

    # 통계
    stats = batch_handler.get_stats()
    print("\nBatch Stats:")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total batches: {stats['total_batches']}")
    print(f"Avg batch size: {stats['avg_batch_size']:.2f}")

    return total_time


async def main():
    """전체 벤치마크 실행"""
    client = VLLMClient()
    batch_handler = BatchHandler(client)

    # 헬스체크
    if not await client.health_check():
        print("❌ vLLM server not available")
        return

    print("✅ vLLM server connected")
    print("=" * 60)

    # 다양한 길이의 요청 생성
    n_requests = 50
    requests = generate_varied_requests(n_requests)

    print(f"\n📊 Generated {n_requests} varied requests")
    print("   Token range: 10-150 tokens")

    # 1. 순차 요청
    single_time, _ = await benchmark_single_requests(client, requests)

    # 2. 동시 요청
    concurrent_time, _ = await benchmark_concurrent_requests(client, requests)

    # 3. 명시적 배치
    batch_time, _ = await benchmark_explicit_batch(client, requests)

    # 4. 배치 핸들러
    handler_time = await benchmark_batch_handler(batch_handler, requests)

    # 결과 요약
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Single requests (est): {single_time:.2f}s (baseline)")
    print(
        f"Concurrent requests:   {concurrent_time:.2f}s ({single_time / concurrent_time:.2f}x faster)"
    )
    print(
        f"Explicit batch:        {batch_time:.2f}s ({single_time / batch_time:.2f}x faster)"
    )
    print(
        f"Batch handler:         {handler_time:.2f}s ({single_time / handler_time:.2f}x faster)"
    )


if __name__ == "__main__":
    asyncio.run(main())
