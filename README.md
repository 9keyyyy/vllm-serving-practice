# vllm-serving-practice

## 현 시스템 아키텍처
로컬 환경(Macbook Pro M4)에서의 테스트를 용이하게 하기 위해 아래처럼 구성 -> 추후 고도화 예정
```
   ┌───────────────────────────────┐
   │           Internet            │
   └───────────────┬───────────────┘
                   │
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐    ┌────────────────────────────────┐
│    Colab     │    │   Kubernetes Cluster           │
│              │    │                                │
│  ┌─────────┐ │    │  ┌──────────────────────────┐  │
│  │  vLLM   │ │    │  │ InferenceService         │  │
│  │  Server │ │    │  │   (llm-api)              │  │
│  │         │ │    │  │                          │  │
│  │ Phi-3   │ │◄───┼──│  FastAPI                 │  │
│  │ (GPU)   │ │    │  │  - Agent                 │  │
│  └─────────┘ │    │  │  - BatchHandler          │  │
│      │       │    │  │  - Monitoring            │  │
│      │       │    │  └──────────────────────────┘  │
│  ┌───▼────┐  │    │              ▲                 │
│  │ ngrok  │  │    │              │                 │
│  └────────┘  │    │        Istio Gateway           │
└──────────────┘    └──────────────┬─────────────────┘
                                   │
                                   ▼
                            User Requests
```


## 폴더 구조
```
llm-serving-practice/
├── .python-version          # Python 3.13
├── pyproject.toml          # uv 의존성 관리
├── README.md
├── .env
├── docker/
│   ├── Dockerfile.api      # FastAPI 서버용
│   └── docker-compose.yml
├── k8s/
│   ├── llm-api-inference.yaml
│   └── monitoring/  # TODO
│       ├── prometheus.yaml
│       └── grafana-dashboard.json
├── src/
│   ├── __init__.py
│   ├── config.py           # 설정 관리
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py      # Pydantic 스키마
│   ├── services/
│   │   ├── __init__.py
│   │   ├── vllm_client.py  # vLLM API 클라이언트
│   │   ├── batch_handler.py # 배치 처리 로직
│   │   └── agent.py         # TODO
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI 앱
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── batch.py
│   │   │   └── health.py
│   │   └── middleware/
│   │       └── metrics.py  # TODO: Prometheus metrics
├── tests/ 
│   ├── __init__.py
│   ├── test_vllm_client.py
│   ├── test_batch.py
│   └── benchmarks/
│       └── batch_performance.py
└── notebooks/
    └── 01_vllm_setup.ipynb      # vllm server (Colab)
```

## vLLM 배치 처리 시스템 설계 및 성능 검증

> vLLM Continuous Batching을 검증하고, 실제 프로덕션 환경을 위한 자동 배치 시스템 설계

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Requests                        │
└────────────────────────┬────────────────────────────────────┘
                         │ add_request()
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     BatchHandler                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Request Queue (deque)                             │     │
│  │  [(request₁, future₁), (request₂, future₂), ...]   │     │
│  └─────────────────────┬──────────────────────────────┘     │
│                        │                                    │
│                        ▼                                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Batch Processor                                   │     │
│  │  • timeout_ms: 100ms 대기                           │     │
│  │  • max_batch_size: 32개 제한                         │     │
│  │  • 조건 충족 시 배치 형성                               │     │
│  └─────────────────────┬──────────────────────────────┘     │
└────────────────────────┼────────────────────────────────────┘
                         │ batch_chat_completion()
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Engine                              │
│                 (Continuous Batching)                       │
│                                                             │
│  • GPU 병렬 처리로 다중 요청 동시 실행                             │
│  • 토큰 길이 무관하게 효율적 처리                                  │
│  • 완료된 요청부터 순차적으로 반환                                  │
└─────────────────────────────────────────────────────────────┘
```


### 테스트 환경
- **요청 수**: 50개
- **토큰 범위**: 10~150 tokens 
- **테스트 시나리오**: 동시 요청 환경 (성능 검증용)

### 처리 방식별 성능 비교

| 처리 방식 | 소요 시간 | Throughput  | 성능 개선 |
|---------|---------|------------|---------|
| 순차 처리 (Baseline) | 129.05s | 0.39 req/s  | 1.0x |
| **동시 요청 ** | 11.41s | 4.38 req/s  | **11.3x** ⬆️ |
| 명시적 배치 | 11.45s | 4.37 req/s  | 11.3x |
| **BatchHandler ** | 17.89s | 2.80 req/s  | **7.2x** ⬆️ |
### 검증 결과 
- vLLM Continuous Batching: 11.3배의 처리량 향상을 제공
- BatchHandler: 7.2배의 처리량 향상을 제공 및 50개 요청을 2회 배치 처리로 완료
  

### 핵심 구현: BatchHandler
실제 프로덕션 환경에서는 요청이 순차적으로 도착하기 때문에, BatchHandler는 이런 환경에서도 자동으로 요청을 수집하여 배치 처리 효과를 얻을 수 있도록 설계
1. **`collections.deque`**: 효율적인 FIFO 큐 구현
2. **시간 기반 배치 형성**: `timeout_ms` 대기로 지연 최소화
3. **크기 제한**: `max_batch_size`로 메모리 효율성 보장
4. **Future 패턴**: 비동기 응답 처리 및 에러 핸들링

**비동기 큐 기반 요청 수집**
```python
async def add_request(self, request: ChatRequest) -> ChatResponse:
    """요청을 큐에 추가하고 배치 처리 결과 대기"""
    future = asyncio.Future()
    self.queue.append((request, future))
    
    # 배치 프로세서 시작
    if not self._processing:
        asyncio.create_task(self._process_batch())
    
    return await future
```

**시간/크기 기반 배치 형성**
```python
async def _process_batch(self):
    """배치 처리 로직"""
    # timeout_ms 대기 후 큐에서 배치 추출
    await asyncio.sleep(self.timeout_ms / 1000)
    
    batch = []
    futures = []
    
    # max_batch_size까지 요청 수집
    while self.queue and len(batch) < self.max_batch_size:
        request, future = self.queue.popleft()
        batch.append(request)
        futures.append(future)
    
    # vLLM으로 배치 전송
    responses = await self.client.batch_chat_completion(batch)
    
    # Future를 통해 각 요청자에게 응답 반환
    for future, response in zip(futures, responses):
        future.set_result(response)
```




### Application Layer (BatchHandler)
여러 개별 요청을 자동으로 수집하여 하나의 배치로 통합

**효과**
- API 호출 횟수 감소 (50회 → 2회)
- 네트워크 오버헤드 최소화
- 개발자는 단일 요청 API 사용 (추상화)

### Engine Layer (vLLM Continuous Batching)
배치로 전달된 요청들을 GPU에서 병렬 처리

**효과**
- 다양한 토큰 길이(10~150) 동시 처리
- 짧은 요청 완료 후에도 긴 요청 계속 실행
- GPU 유휴 시간 최소화로 처리량 극대화

### 통합 효과
```
BatchHandler
  └─→ 요청 자동 수집 및 배치 형성
        └─→ vLLM Engine
              └─→ GPU 병렬 처리 (Continuous Batching)
                    └─→ 11.3배 성능 향상 달성
```




## TODO
- [x] vLLM 서버 세팅
- [x] FastAPI 서버 세팅 및 모델 연동 확인
- [x] KServe 서빙
- [x] continuous batching/batch handler 성능 평가
- [ ] Prometheus + Grafana 모니터링
- [ ] LangChain Agent 구현
- [ ] 클라우드 환경 GPU 활용 vLLM 서버 세팅