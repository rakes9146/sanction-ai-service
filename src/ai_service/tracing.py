"""
tracing.py - LangSmith tracing (optional, off by default).

WHAT: when `langsmith.enabled: true` in config/application.yml, every alert shows up in LangSmith as one trace:
        run_alert (the whole graph)  ->  one child per node  ->  the OpenAI calls with prompt, answer, tokens and latency.
WHY:  you can open a slow or odd result and see exactly which step and which prompt produced it.

How it works (nothing to code in the nodes):
  * LangGraph and the `langsmith` library read LANGSMITH_* environment variables. We keep the values in the property file,
    so `setup_tracing` copies them into the process environment ONCE at start-up, before the first alert.
  * `wrap_client` wraps the OpenAI client so each model call becomes its own span.

PRIVACY: prompts contain the payment text. This POC uses synthetic data only. With real data set `langsmith.hide_inputs` and
`langsmith.hide_outputs` to true (traces then show the shape and timings but not the content), or point `langsmith.endpoint` at a
self-hosted LangSmith inside your network.
"""
from __future__ import annotations

import os

import structlog

from ai_service.config import Settings

log = structlog.get_logger()


def setup_tracing(settings: Settings) -> bool:
    """Switch tracing on or off for this process. Returns True when traces will be sent."""
    if not settings.langsmith_enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_HIDE_INPUTS"] = str(settings.langsmith_hide_inputs).lower()
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = str(settings.langsmith_hide_outputs).lower()
    log.info("langsmith tracing on", project=settings.langsmith_project, endpoint=settings.langsmith_endpoint,
             hide_inputs=settings.langsmith_hide_inputs, hide_outputs=settings.langsmith_hide_outputs)
    return True


def flush_traces(settings: Settings) -> None:
    """Send what is still queued. LangSmith ships traces from a background thread in batches, so without this a stop or restart loses the last few."""
    if not settings.langsmith_enabled:
        return
    try:
        from langsmith.run_trees import get_cached_client

        get_cached_client().flush()
        log.info("langsmith traces flushed")
    except Exception as e:                     # noqa: BLE001 - never let tracing break a shutdown
        log.warning("langsmith flush failed", error=str(e))


def wrap_client(client, settings: Settings):
    """Return the OpenAI client wrapped for tracing, or the same client when tracing is off."""
    if not settings.langsmith_enabled:
        return client
    from langsmith.wrappers import wrap_openai

    return wrap_openai(client)


def run_config(envelope: dict) -> dict:
    """Name, tags and metadata for one traced alert, so you can search LangSmith by payment_id or request_id."""
    payload = envelope.get("payload") or {}
    return {
        "run_name": "ai_review_alert",
        "tags": ["sanction-ai-service", str(payload.get("review_type") or payload.get("type") or "alert")],
        "metadata": {"payment_id": envelope.get("payment_id", ""), "request_id": envelope.get("screening_request_id", ""),
                     "event_id": envelope.get("event_id", "")},
    }
