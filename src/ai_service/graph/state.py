"""
state.py - the "notebook" that travels through the graph, and the things the graph depends on.

WHAT: `AlertState` is a plain dictionary with known keys (a TypedDict). Every node reads what it needs from it and returns ONLY the
      keys it wants to change; LangGraph merges that into the notebook before the next node runs.
WHY:  one visible place that lists everything the pipeline knows about one alert. If you wonder "where does X come from?", look
      for the node that returns key X.

New to Python? A TypedDict is a dict whose keys and value types are written down. `total=False` means "keys may be missing"
(early nodes run before later keys exist), so always read with state.get("key", default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, TypedDict

from ai_service.config import Rules, Settings
from ai_service.llm.base import LLM
from ai_service.models import AiRequestPayload, L1Output, L2Output
from ai_service.rag.retriever import PolicyRetriever


class AlertState(TypedDict, total=False):
    # ---- input (set by the caller before the graph starts) ----
    envelope: dict                  # the whole ai.review.request.v1 event
    request: AiRequestPayload       # its payload, validated
    payment_id: str
    request_id: str

    # ---- prepare ----
    texts: list[dict]               # payment text shown to the model: NAME / ADDRESS / FREE_TEXT fields
    all_texts: list[str]            # every text of the message; evidence quotes are checked against these
    floors: list[str]               # safety floors known so far (grows as nodes find more)
    security_event: bool            # injection found (PA-03 regex or the AI's L1-09)
    attribute_table: list[dict]     # PA-02: one row per name hit: dob / nationality / city / identifiers = MATCH|MISMATCH|UNKNOWN

    # ---- AI L1 ----
    l1_plan: list[str]              # which L1 questions are switched on
    l1_raw: Optional[L1Output]      # what the model said
    l1: Optional[L1Output]          # what is left after the evidence check (RC-02)
    dropped_evidence: list[dict]    # answers removed because their quote is not in the message

    # ---- re-screen (matching stays in the Java screening service) ----
    rescreen_status: str            # NOT_NEEDED | COMPLETED | TIMEOUT
    rescreen_items: list[dict]
    rescreen_hits: list[dict]       # new hits found by re-screening what the AI extracted

    # ---- score ----
    hit_scores: list[dict]
    alert_score: float
    raw_band: str                   # band from the score alone
    effective_band: str             # strictest of raw band and all floors
    code_rechecks: dict

    # ---- AI L2 ----
    clauses: list[dict]             # the retrieved policy clauses
    prior_decisions: list[dict]
    l2: Optional[L2Output]
    gate: dict                      # result of the final gate

    # ---- outcome ----
    proposed_route: str             # AUTO_CLEARED | L3
    qa_sample: bool
    status: str                     # COMPLETED | FAILED
    failed: bool                    # True once something went wrong: every edge then jumps to `emit`
    errors: list[str]
    ai_level_reached: Optional[str]
    usage: list[dict]               # one entry per LLM call
    timings: dict                   # node name -> milliseconds
    audit: list[dict]               # audit events to publish: {"action": ..., "details": {...}}
    result: dict                    # the finished ai.review.result payload


# ---- re-screening: the graph asks "something outside" to screen the names/places/ids the AI found ---------------------
@dataclass
class RescreenOutcome:
    ok: bool                        # False = no answer in time
    hits: list[dict]                # [{item_id, rule_id, severity, similarity, watchlist_entry_id, floor_hint, matched_text}]
    floor_hints: list[str]
    reason: str = ""


class Rescreener(Protocol):
    """Implemented by messaging/rescreen.py (Kafka) and by a stub in the tests."""

    def rescreen(self, payment_id: str, request_id: str, traceparent: Optional[str], items: list[dict]) -> RescreenOutcome: ...


class Repo(Protocol):
    """The few database questions the graph asks. Implemented by repo.py (MongoDB and in-memory)."""

    def get_flag(self, name: str) -> bool: ...
    def prior_decisions(self, party_names: list[str], entry_ids: list[str], limit: int = 3) -> list[dict]: ...


@dataclass
class Deps:
    """Everything the nodes need, handed over once when the graph is built (so nodes never create their own connections)."""

    settings: Settings
    rules: Rules
    llm: LLM
    retriever: PolicyRetriever
    rescreener: Rescreener
    repo: Repo
