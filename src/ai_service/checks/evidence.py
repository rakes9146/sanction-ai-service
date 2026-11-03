"""
evidence.py - RC-02 "evidence is real".

WHAT: checks that a quote the AI gave is an EXACT piece of the payment text.
WHY:  the AI may not rely on facts that are not in the message. If it invents or "tidies" a quote, we drop that answer
      (it becomes UNKNOWN) instead of believing it.

The only tolerance: runs of white space (spaces, tabs, new lines) count as one space, because the AI sees "VIA TEHRAN\\nSTREET"
and may write "VIA TEHRAN STREET". Case, spelling and punctuation must match exactly.
"""
from __future__ import annotations


def collapse(text: str) -> str:
    """'VIA  TEHRAN\\nSTREET' -> 'VIA TEHRAN STREET'"""
    return " ".join(text.split())


def evidence_found(evidence: str, texts: list[str]) -> bool:
    """True if `evidence` is a substring of one of the message texts (or of all of them joined with a space)."""
    needle = collapse(evidence)
    if not needle:
        return False
    collapsed = [collapse(t) for t in texts]
    if any(needle in t for t in collapsed):
        return True
    # the AI may quote across two neighbouring fields, for example the two lines of field 70
    return needle in " ".join(collapsed)
