"""
qa_sampling.py - floor G-10: quality-check a share of the payments the AI cleared.

WHAT: decides, from the payment id only, whether a cleared payment is ALSO queued for a human to look at afterwards.
WHY:  the money has already moved, but a person regularly checks that the AI clears the right payments.
      Using a hash of the payment id (instead of a random number) means replaying the same payment gives the same answer.
"""
from __future__ import annotations

import hashlib


def is_qa_sample(payment_id: str, percent: int) -> bool:
    """True for roughly `percent` out of every 100 payment ids."""
    bucket = int(hashlib.sha256(payment_id.encode("utf-8")).hexdigest(), 16) % 100
    return bucket < percent
