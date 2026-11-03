"""
main.py - starts the service: a tiny web server (only /health) plus the Kafka reader threads.

Run it:   uv run python -m ai_service.main          (reads .env; see .env.example)
Check it: http://localhost:8083/health

FastAPI 'lifespan' = code that runs once at start-up (before `yield`) and once at shutdown (after it).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ai_service.config import Settings
from ai_service.logging_setup import setup_logging
from ai_service.messaging.service import AiService

log = structlog.get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    service_holder: dict[str, AiService] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        service = AiService(settings)           # connects to Mongo / Kafka, loads the policy clauses, builds the graph
        service.start()
        service_holder["service"] = service
        log.info("ai service started", llm_provider=settings.llm_provider, embedder=settings.embedder, port=settings.server_port)
        yield
        service.stop()

    app = FastAPI(title="sanction-ai-service", lifespan=lifespan)

    @app.get("/health")
    def health() -> JSONResponse:
        service = service_holder.get("service")
        body = service.health() if service else {"status": "STARTING"}
        return JSONResponse(body, status_code=200 if body["status"] == "UP" else 503)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("ai_service.main:app", host="0.0.0.0", port=Settings().server_port)
