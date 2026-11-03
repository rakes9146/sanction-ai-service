"""
retriever.py - RETRIEVAL: choose the (at most) 5 policy clauses to show the AI for one alert.

Why not "just do a vector search"?
    A vector search returns the clauses that SOUND most similar. But some clauses must be shown every time (for example SCP-8.1:
    the AI may only clear below 0.35), and some are required for a given kind of hit (a banned BIC needs SCP-5.1). A rule that
    matters must never depend on an embedding score. So retrieval is a LADDER, strongest rule first, vector search last:

        step 1  PINNED    clauses flagged pinned in the corpus. Always included.                  (metadata filter)
        step 2  TAGGED    clauses whose applies_to_hit_types matches this alert (hit rule ids,    (metadata filter)
                          plus labels such as STREET_NAME or INJECTION), and whose formats fit.
                          Clauses matching MORE tags come first.
        step 3  SEMANTIC  only if slots are left: vector search with a text describing the alert. (vector search)

    The result is at most `k` (default 5) clauses, each marked with WHY it was chosen: pinned | tagged | semantic.

Why only 5?  Short prompts are cheaper, faster and easier for the model to follow. Change K below to tune it.
The AI may cite ONLY the ids returned here. The final gate (graph/nodes.py) rejects any other id: that is the guard against the
model inventing a policy clause.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ai_service.rag.embedder import Embedder
from ai_service.rag.store import PolicyStore

K = 5   # total clauses handed to the AI


@dataclass
class Clause:
    clause_id: str
    title: str
    text: str
    version: str
    why: str                      # pinned | tagged | semantic
    score: Optional[float] = None  # only for semantic hits

    def as_dict(self) -> dict:
        return {"clause_id": self.clause_id, "title": self.title, "text": self.text, "version": self.version, "why": self.why}


class PolicyRetriever:
    def __init__(self, store: PolicyStore, embedder: Embedder, k: int = K) -> None:
        self._store = store
        self._embedder = embedder
        self._k = k

    # ---- the main entry point, used by the graph ----------------------------------------------------------------------
    def retrieve(self, tags: set[str], fmt: Optional[str], query_text: str) -> list[Clause]:
        """
        tags        what this alert is about: rule ids of the hits (S2-R04 ...) and labels (STREET_NAME, INJECTION ...)
        fmt         message format (NEFT, RTGS, MT103, PACS008); a clause listing formats only applies to those
        query_text  plain sentence describing the alert, used only for step 3
        """
        chosen: list[Clause] = []

        # step 1: PINNED - always in. Sorted so the order is stable.
        for payload in sorted(self._store.pinned(), key=lambda p: p["clause_id"]):
            chosen.append(self._clause(payload, "pinned"))

        # step 2: TAGGED - the store filters by tag; we then drop clauses for other formats and rank by how many tags match.
        candidates = [p for p in self._store.tagged(sorted(tags)) if self._format_ok(p, fmt)]
        candidates.sort(key=lambda p: (-len(tags & set(p["applies_to_hit_types"])), p["clause_id"]))
        for payload in candidates:
            if len(chosen) >= self._k:
                break
            if payload["clause_id"] not in {c.clause_id for c in chosen}:
                chosen.append(self._clause(payload, "tagged"))

        # step 3: SEMANTIC - fill whatever is left with the clauses closest in meaning to the description of the alert.
        free = self._k - len(chosen)
        if free > 0 and query_text.strip():
            vector = self._embedder.embed([query_text])[0]
            for payload, score in self._store.nearest(vector, free, [c.clause_id for c in chosen]):
                chosen.append(self._clause(payload, "semantic", score))
        # the loops above never add past k, and pinned clauses are never cut, so `chosen` is the final answer
        return chosen

    # ---- helpers for the final gate and for scripts/ask_policy.py ------------------------------------------------------
    def known_clause_ids(self) -> set[str]:
        """Every clause id in the CURRENT policy version. The gate checks cited ids against this."""
        return {p["clause_id"] for p in self._store.all_payloads().values()}

    def version(self) -> str:
        payloads = list(self._store.all_payloads().values())
        return payloads[0]["version"] if payloads else "NO_POLICY_LOADED"

    def ask(self, question: str, n: int = 3) -> list[Clause]:
        """Pure semantic search (no pinned, no tags). Used by scripts/ask_policy.py to look at retrieval in isolation."""
        vector = self._embedder.embed([question])[0]
        return [self._clause(p, "semantic", s) for p, s in self._store.nearest(vector, n, [])]

    # ---- small private helpers ----------------------------------------------------------------------------------------
    @staticmethod
    def _format_ok(payload: dict, fmt: Optional[str]) -> bool:
        formats = payload.get("applies_to_formats") or []
        return not formats or (fmt or "").upper() in [f.upper() for f in formats]

    @staticmethod
    def _clause(payload: dict, why: str, score: Optional[float] = None) -> Clause:
        return Clause(payload["clause_id"], payload["title"], payload["text"], payload["version"], why, score)
