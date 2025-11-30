# vllm-serving-practice

### 현 시스템 아키텍처
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


### 폴더 구조
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
├── tests/ # TODO
│   ├── __init__.py
│   ├── test_vllm_client.py
│   ├── test_batch.py
│   └── benchmarks/
│       └── batch_performance.py
└── notebooks/
    └── 01_vllm_setup.ipynb      # vllm server (Colab)
```


### TODO
- [x] vLLM 서버 세팅
- [x] FastAPI 서버 세팅 및 모델 연동 확인
- [x] KServe 서빙
- [ ] continuos batch 성능 평가
- [ ] Prometheus + Grafana 모니터링
- [ ] LangChain Agent 구현
- [ ] 클라우드 환경 GPU 활용 vLLM 서버 세팅