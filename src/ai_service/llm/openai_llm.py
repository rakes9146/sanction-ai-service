"""
openai_llm.py - the real model, through the OpenAI Python SDK with STRUCTURED OUTPUT.

WHAT: `read` (AI L1) and `reason` (AI L2) each make one call. We pass a Pydantic class (L1Output / L2Output) as `response_format`;
      OpenAI then forces the answer to match that shape, and the SDK hands back a ready Pydantic object.
WHY:  structured output means no parsing of free text and no "creative" extra fields. If the shape is still wrong (a refusal, a cut-off
      answer) we raise LLMSchemaError and the graph retries once.

Retries: network trouble / rate limits / 5xx are retried 3 times, 0.5 s apart (with a little random jitter), then LLMUnavailable.
Settings used (config/application.yml): openai.api_key, openai.base_url (local server), OPENAI_MODEL_L1, OPENAI_MODEL_L2, LLM_TEMPERATURE (default 0), LLM_TIMEOUT_S. No model name is in the code.
"""
from __future__ import annotations

import time

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed, wait_random

from ai_service.config import Settings
from ai_service.llm.base import L1Request, L2Request, LLM
from ai_service.llm.prompts import render_l1, render_l2
from ai_service.tracing import wrap_client
from ai_service.models import L1Output, L2Output, LLMSchemaError, LLMUnavailable, Usage

# errors that are worth trying again
_TRANSIENT = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)


def _temperature_arg(model: str, temperature: float) -> dict:
    """gpt-5 and the o-series reasoning models only accept their default temperature: sending 0 gives HTTP 400, so we leave it out."""
    reasoning = model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
    return {} if reasoning else {"temperature": temperature}


class OpenAILLM(LLM):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = wrap_client(                                                # wrapped only when langsmith.enabled
            OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None, timeout=settings.llm_timeout_s,
                   max_retries=0),                                             # we do the retrying
            settings)

    def read(self, request: L1Request) -> tuple[L1Output, Usage]:
        system, user = render_l1(request)
        return self._call("L1", self._settings.openai_model_l1, system, user, L1Output)

    def reason(self, request: L2Request) -> tuple[L2Output, Usage]:
        system, user = render_l2(request)
        return self._call("L2", self._settings.openai_model_l2, system, user, L2Output)

    # ---------------------------------------------------------------------------------------------------------------
    def _call(self, step: str, model: str, system: str, user: str, schema: type[BaseModel]):
        started = time.perf_counter()
        attempts = 0
        retrying = Retrying(stop=stop_after_attempt(3), wait=wait_fixed(0.5) + wait_random(0, 0.25),
                            retry=retry_if_exception_type(_TRANSIENT), reraise=True)
        try:
            for attempt in retrying:
                with attempt:
                    attempts = attempt.retry_state.attempt_number
                    completion = self._client.chat.completions.parse(
                        model=model,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        response_format=schema,
                        **_temperature_arg(model, self._settings.llm_temperature),
                    )
        except _TRANSIENT as e:
            raise LLMUnavailable(f"{type(e).__name__} after {attempts} attempts") from e
        except openai.APIError as e:                     # any other API error (bad request, auth ...) cannot be fixed by retrying
            raise LLMUnavailable(f"{type(e).__name__}: {e}") from e
        except ValidationError as e:                     # the SDK could not fit the answer into our Pydantic class
            raise LLMSchemaError(str(e)[:300]) from e

        message = completion.choices[0].message
        if message.refusal or message.parsed is None:
            raise LLMSchemaError("the model refused or returned no parsable answer")
        usage = Usage(step=step, model=model, prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
                      completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
                      latency_ms=int((time.perf_counter() - started) * 1000), retries=max(attempts - 1, 0))
        return message.parsed, usage
