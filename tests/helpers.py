"""
helpers.py - shared test tools: build the service's dependencies without Kafka / MongoDB / OpenAI, and build alert requests.

Everything here is offline:
    * the vector store lives in memory (QdrantClient(":memory:")) and the 14 policy clauses are ingested with the HashEmbedder
    * the model is a ScriptedLLM: the test says exactly what the "AI" answers
    * re-screening is a StubRescreener: the test says what the Java screening service would answer
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

from ai_service.config import Rules, Settings
from ai_service.graph.builder import build_graph, run_alert
from ai_service.graph.state import Deps, RescreenOutcome
from ai_service.llm.base import LLM
from ai_service.rag.embedder import HashEmbedder
from ai_service.rag.ingest import ingest
from ai_service.rag.retriever import PolicyRetriever
from ai_service.rag.store import PolicyStore
from ai_service.repo import MemoryRepo

DATA = Path(__file__).parent / "data"


def load_event(prefix: str) -> dict:
    """Load one of the sample events in tests/data by the start of its file name (for example '04')."""
    path = next(DATA.glob(f"{prefix}_*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


class StubRescreener:
    """Pretends to be the Java screening service. Records what it was asked."""

    def __init__(self, outcome: Optional[RescreenOutcome] = None) -> None:
        self.outcome = outcome or RescreenOutcome(ok=True, hits=[], floor_hints=[])
        self.calls: list[list[dict]] = []

    def rescreen(self, payment_id, request_id, traceparent, items):
        self.calls.append(items)
        return self.outcome


def make_deps(llm: LLM, rescreener: Optional[StubRescreener] = None, repo: Optional[MemoryRepo] = None, **settings) -> Deps:
    s = Settings(_env_file=None, llm_provider="fake", embedder="hash", **settings)      # _env_file=None: ignore any local .env
    embedder = HashEmbedder()
    store = PolicyStore()                                                              # in memory
    ingest(store, embedder)
    return Deps(settings=s, rules=Rules.load(), llm=llm, retriever=PolicyRetriever(store, embedder),
                rescreener=rescreener or StubRescreener(), repo=repo or MemoryRepo())


def run(deps: Deps, event: dict) -> dict:
    """Run one alert through the real graph and return the final state."""
    return run_alert(build_graph(deps), deps, event)


# ---- building requests -------------------------------------------------------------------------------------------------
def tc02_event() -> dict:
    """Sample 04: NEFT, fuzzy name 'Mohd Abdul Hafiz' vs listed 'Muhammad Abdul Hafeez'. Different person: DOB, nationality, city all differ."""
    return load_event("04")


def free_text_event(text: str, rule_id: str, matched: Optional[str], floors: list[str], fmt: str = "MT103", severity: str = "SOFT",
                    name: Optional[str] = None, customer: bool = False, injection: bool = False, similarity: Optional[float] = None,
                    list_entry: Optional[dict] = None, entry_id: str = "GAZ-CITY-TEHRAN") -> dict:
    """
    A request with ONE hit and ONE free-text field (field f1, path '70'). Start from sample 04 and replace what differs.
    Use `name` for a name hit (then the hit points at field f2, a NAME field).
    """
    event = copy.deepcopy(load_event("04"))
    payload = event["payload"]
    payload["format"] = fmt
    fields = [{"field_id": "f1", "path": "70", "kind": "FREE_TEXT", "role": "REMITTANCE", "value": text}]
    parties = []
    hit = {"hit_id": "h1", "rule_id": rule_id, "severity": severity, "field_id": "f1", "field_path": "70", "original_value": text,
           "matched_text": matched, "similarity": similarity, "watchlist_entry_id": entry_id, "watchlist_version": "POC-WL-2026-10-01.1",
           "list_entry": list_entry, "floor_hint": None}
    if name:
        fields.append({"field_id": "f2", "path": "Cdtr/Nm", "kind": "NAME", "role": "CREDITOR", "value": name})
        parties = [{"role": "CREDITOR", "name": name, "account": "****1", "identifiers": [],
                    "address": {"lines": [], "city": "Sharjah", "region": None, "postal_code": None, "country": "AE"}}]
        hit.update({"field_id": "f2", "field_path": "Cdtr/Nm", "original_value": name, "matched_text": name})
    payload["alert"]["hits"] = [hit]
    payload["alert"]["floors_so_far"] = floors
    payload["alert"]["pre_ai_checks"]["PA-03"] = {"injection_found": injection, "patterns": ["INJ-01"] if injection else []}
    payload["untrusted_message"]["parsed_message"] = {"format": fmt, "parties": parties, "agents": [], "screenable_fields": fields,
                                                        "free_text": [{"field_id": "f1", "path": "70", "value": text}], "absent_roles": []}
    if not customer:
        payload["customer"] = {"found": False}
    return event
