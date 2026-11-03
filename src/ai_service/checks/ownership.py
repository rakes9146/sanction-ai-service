"""
ownership.py - L2-03: "does the text show ownership or control by a listed party?"

WHAT: the AI lists ownership statements it finds ("payment for X Ltd, 60% owned by Rostek Holdings"). CODE then
      (a) reads the percentage, and (b) checks whether the OWNER is a listed party.
WHY:  language models are unreliable at arithmetic and at deciding who is on a list. So the AI only extracts; code judges.
      Any statement whose owner looks like a listed name stops the AI from clearing the payment.
"""
from __future__ import annotations

import re
from typing import Optional

from rapidfuzz.distance import JaroWinkler

from ai_service.models import OwnershipStatement


def parse_percent(text: Optional[str]) -> Optional[float]:
    """'60%' -> 60.0 ; 'sixty percent' -> None (we do not guess)."""
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    return float(m.group(1)) if m else None


def listed_owners(statements: list[OwnershipStatement], listed_names: list[str], threshold: float = 0.85) -> list[dict]:
    """Statements whose owner is close to a listed name (same Jaro-Winkler idea as the screening service, kept simple)."""
    found = []
    for s in statements:
        for name in listed_names:
            if JaroWinkler.similarity(s.owner.upper(), name.upper()) >= threshold:
                found.append({"owner": s.owner, "owned": s.owned, "percent": parse_percent(s.percent_text), "listed_name": name})
                break
    return found
