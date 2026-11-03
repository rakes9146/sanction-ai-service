"""
OPTIONAL smoke test against the real OpenAI API. Skipped unless you ask for it:

    RUN_LIVE_LLM=1 uv run pytest tests/e2e/test_live_openai.py -s

Needs LLM_PROVIDER=openai, OPENAI_API_KEY, OPENAI_MODEL_L1, OPENAI_MODEL_L2 (and EMBEDDER=openai + OPENAI_EMBEDDING_MODEL if you want real embeddings)
in your environment or .env. It makes two tiny calls (a few cents). It is NOT part of the normal test run and CI never runs it.
It checks the plumbing (structured output parses into our models, usage is recorded) - not the quality of the model's labels.
"""
import os

import pytest

from ai_service.config import Settings
from ai_service.llm.base import L1Request, L2Request
from ai_service.llm.openai_llm import OpenAILLM
from ai_service.models import L1Output, L2Output

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1 to call the real API")

TEXT = "FREIGHT CHARGES CONTAINER 4021 VIA TEHRAN STREET HUB DUBAI"


def test_l1_and_l2_structured_output_with_the_real_model():
    settings = Settings(llm_provider="openai")
    settings.check_openai_ready()
    llm = OpenAILLM(settings)
    texts = [{"field_id": "f1", "path": "70", "kind": "FREE_TEXT", "value": TEXT}]
    hits = [{"hit_id": "h1", "rule_id": "S2-R04", "original_value": TEXT, "matched_text": "TEHRAN", "field_path": "70"}]

    l1, usage1 = llm.read(L1Request(checks=["L1-01", "L1-05", "L1-09"], hits=hits, texts=texts))
    assert isinstance(l1, L1Output) and usage1.prompt_tokens > 0 and usage1.latency_ms > 0
    print("L1:", l1.model_dump(exclude_none=True, exclude_defaults=True))

    clauses = [{"clause_id": "SCP-6.2", "title": "Street name", "text": "A place word that is part of a street name is not by itself a sanctioned location."}]
    l2, usage2 = llm.reason(L2Request(questions=["L2-01", "L2-05", "L2-06"], hit_types=["S2-R04"], facts=l1.model_dump(exclude_none=True, exclude_defaults=True),
                                      clauses=clauses, texts=texts, alert_score=0.3, floors=["G-03"]))
    assert isinstance(l2, L2Output) and usage2.completion_tokens > 0
    print("L2:", l2.model_dump(exclude_none=True))
