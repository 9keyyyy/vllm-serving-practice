from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # vLLM 서버 설정
    vllm_base_url: str = "http://localhost:8000"
    vllm_api_key: str = ""
    vllm_model: str = "microsoft/Phi-3-mini-4k-instruct"

    # 배치 처리 설정
    batch_max_size: int = 32  # vLLM의 continuous batching 활용
    batch_timeout_ms: int = 100  # 100ms 대기

    # API 서버 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 4

    # 모델 설정
    max_tokens: int = 512
    temperature: float = 0.7

    # 모니터링
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Kubernetes (Prod)
    kserve_endpoint: Optional[str] = None
    use_kserve: bool = False


settings = Settings()
