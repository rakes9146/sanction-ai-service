FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src src
COPY config config
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN useradd --system --uid 10001 --create-home ai && mkdir -p /app/data && chown ai /app/data
WORKDIR /app
COPY --from=build --chown=ai /app /app
USER ai
ENV PATH="/app/.venv/bin:$PATH" QDRANT_PATH=/app/data/qdrant SERVER_PORT=8083
EXPOSE 8083
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8083/health').status == 200 else 1)"
CMD ["python", "-m", "ai_service.main"]
