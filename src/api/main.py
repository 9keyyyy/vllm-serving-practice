import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from src.api.middleware.metrics import MetricsMiddleware
from src.api.routes import batch, chat, health
from src.config import settings
from src.services.batch_handler import BatchHandler
from src.services.vllm_client import VLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing vLLM client...")

    app.state.vllm_client = VLLMClient()
    app.state.batch_handler = BatchHandler(app.state.vllm_client)

    # Health check
    if await app.state.vllm_client.health_check():
        logger.info("✅ vLLM server connected")
    else:
        logger.warning("⚠️ vLLM server not available")

    yield

    # 종료
    logger.info("Shutting down...")


app = FastAPI(
    title="LLM Serving Platform",
    description="Production-ready LLM serving with vLLM batch processing",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
# if settings.enable_metrics:
#     app.add_middleware(MetricsMiddleware)

app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(batch.router, prefix="/api/v1", tags=["batch"])
app.include_router(health.router, tags=["health"])


@app.get("/")
async def root():
    return {
        "name": "LLM Serving Platform",
        "version": "0.1.0",
        "vllm_model": settings.vllm_model,
    }
