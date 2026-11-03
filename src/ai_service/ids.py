"""
ids.py - small helpers that make ids and timestamps in the format the shared contract asks for.

WHAT: UUIDv7 (event ids), ULID (aud_ ids), a deterministic id for results, and the RFC 3339 timestamp with milliseconds.
WHY:  every service must create ids in the same shape, and the result id must be the SAME if the same request is processed twice.
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def now_iso() -> str:
    """'2026-10-17T17:52:48.514Z' - UTC, always 3 digits of milliseconds."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def uuid7() -> str:
    """UUIDv7: 48 bits of unix milliseconds, version 7, variant 10, the rest random. Sorts by time."""
    millis = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    value = (millis << 80) | (0x7 << 76) | (((rand >> 62) & 0xFFF) << 64) | (0b10 << 62) | (rand & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


def ulid() -> str:
    """26 characters: 10 of time + 16 random (Crockford base32). Used for 'aud_<ULID>' audit ids."""
    millis = int(time.time() * 1000)
    chars = []
    for _ in range(10):
        chars.append(_CROCKFORD[millis & 31])
        millis >>= 5
    time_part = "".join(reversed(chars))
    rand_part = "".join(_CROCKFORD[b & 31] for b in os.urandom(16))
    return time_part + rand_part


def result_event_id(request_event_id: str) -> str:
    """
    The id of our answer, derived from the id of the request.
    Why: if Kafka delivers the same request twice and we answer twice, both answers carry the SAME event_id,
    so the orchestrator (which ignores event ids it has already seen) treats the second one as a duplicate.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "ai-result:" + request_event_id))
