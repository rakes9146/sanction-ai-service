"""
base.py - the LLM interface: what the graph needs from "the AI", and nothing more.

WHAT: two request shapes (L1Request, L2Request) and an abstract class LLM with two methods, `read` (AI L1) and `reason` (AI L2).
WHY:  the graph only talks to this interface. So the SAME graph runs with
        - OpenAILLM    the real model              (llm/openai_llm.py)
        - FakeLLM      a rule-based demo stand-in  (llm/fake_llm.py)   <- not AI, no key needed
        - ScriptedLLM  canned answers for tests    (llm/fake_llm.py)
      Changing provider never touches the graph.

New to Python? `ABC` + `@abstractmethod` mean "a class that inherits from LLM MUST write these two methods".
`@dataclass` writes the boring __init__ for a class that only holds data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ai_service.models import L1Output, L2Output, Usage


@dataclass
class L1Request:
    """Everything AI L1 gets. `texts` is the payment text and is UNTRUSTED: it goes to the model only inside the data block."""

    checks: list[str]                  # the L1 questions switched on, for example ["L1-01", "L1-05", "L1-09"]
    hits: list[dict]                   # compact hits: hit_id, rule_id, original_value, matched_text, field_path
    texts: list[dict]                  # [{"field_id": "f9", "path": "remittance_info", "kind": "FREE_TEXT", "value": "..."}]


@dataclass
class L2Request:
    """Everything AI L2 gets: verified facts, the policy clauses, our customer (if any) and earlier decisions."""

    questions: list[str]               # L2 questions to answer, for example ["L2-01", "L2-05", "L2-06"]
    hit_types: list[str]               # rule ids of the hits
    facts: dict                        # the VERIFIED L1 answers (after the evidence check)
    clauses: list[dict]                # the (at most 5) retrieved policy clauses: clause_id, title, text
    texts: list[dict]                  # the payment text (untrusted)
    customer: Optional[dict] = None    # KYC summary, only if customer.found
    prior_decisions: list[dict] = field(default_factory=list)   # up to 3 earlier decisions on the same party + list entry
    alert_score: float = 0.0           # for the model's context only; it cannot change it
    floors: list[str] = field(default_factory=list)


class LLM(ABC):
    @abstractmethod
    def read(self, request: L1Request) -> tuple[L1Output, Usage]:
        """AI L1: read and label. Raises LLMSchemaError (bad shape) or LLMUnavailable (cannot reach the model)."""

    @abstractmethod
    def reason(self, request: L2Request) -> tuple[L2Output, Usage]:
        """AI L2: use the policy clauses to explain and recommend. Same errors as `read`."""
