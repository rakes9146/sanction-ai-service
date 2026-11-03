"""
repo.py - all database access in one file (MongoDB, `ai_db`).

Collections:
    ai_runs            one document per alert: the request id, the hits' party names and list entries, every node's output,
                       the versions, token usage, timings and the final result. Used for audit, replay and decision memory.
    decision_memory    "this party + this list entry was decided as RELEASE/REJECT/BLOCK on <date>". Feeds AI L2-04 (prior alerts).
                       Built by joining the final payment.status events to the ai_runs row of the same payment.
    runtime_flags      {_id: "ai_kill_switch", on: true/false}  - an operator can flip it without a restart.
    processed_events   {_id: "<consumer>:<event_id>"}  - makes every consumer safe against duplicate deliveries.

Two classes with the same methods:
    MongoRepo   the real thing (pymongo, plain blocking calls).
    MemoryRepo  the same in plain Python dicts: used by unit tests and by scripts/run_sample.py (no database needed).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError

from ai_service.config import Settings


def _party_key(name: str) -> str:
    """A name reduced to letters and digits, upper case: 'Ravi  Kumar' and 'RAVI KUMAR' become the same key."""
    return "".join(ch for ch in (name or "").upper() if ch.isalnum())


class MemoryRepo:
    def __init__(self) -> None:
        self.flags: dict[str, bool] = {}
        self.runs: dict[str, dict] = {}
        self.decisions: list[dict] = []
        self.processed: set[str] = set()

    # flags
    def get_flag(self, name: str) -> bool:
        return self.flags.get(name, False)

    def set_flag(self, name: str, on: bool) -> None:
        self.flags[name] = on

    # runs
    def save_run(self, run: dict) -> None:
        self.runs[run["_id"]] = run

    def find_run(self, event_id: str) -> Optional[dict]:
        """The run for one request, found by the event id of the request (that id is the document id)."""
        return self.runs.get(event_id)

    def find_run_by_payment(self, payment_id: str) -> Optional[dict]:
        found = [r for r in self.runs.values() if r.get("payment_id") == payment_id]
        return found[-1] if found else None

    # decision memory
    def save_decision(self, decision: dict) -> None:
        self.decisions.append(decision)

    def prior_decisions(self, party_names: list[str], entry_ids: list[str], limit: int = 3) -> list[dict]:
        keys = {_party_key(n) for n in party_names}
        matches = [d for d in self.decisions if d["party_key"] in keys and d["watchlist_entry_id"] in set(entry_ids)]
        matches.sort(key=lambda d: d["decided_at"], reverse=True)
        return matches[:limit]

    # idempotency
    def is_processed(self, consumer: str, event_id: str) -> bool:
        return f"{consumer}:{event_id}" in self.processed

    def mark_processed(self, consumer: str, event_id: str) -> None:
        self.processed.add(f"{consumer}:{event_id}")


class MongoRepo:
    def __init__(self, settings: Settings) -> None:
        self._client = MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000)
        self._db = self._client[settings.mongo_db]
        self._db.ai_runs.create_index([("payment_id", ASCENDING)])
        self._db.decision_memory.create_index([("party_key", ASCENDING), ("watchlist_entry_id", ASCENDING), ("decided_at", DESCENDING)])

    def ping(self) -> bool:
        self._client.admin.command("ping")
        return True

    # flags
    def get_flag(self, name: str) -> bool:
        doc = self._db.runtime_flags.find_one({"_id": name})
        return bool(doc and doc.get("on"))

    def set_flag(self, name: str, on: bool) -> None:
        self._db.runtime_flags.update_one({"_id": name}, {"$set": {"on": on}}, upsert=True)

    # runs
    def save_run(self, run: dict) -> None:
        self._db.ai_runs.replace_one({"_id": run["_id"]}, run, upsert=True)

    def find_run(self, event_id: str) -> Optional[dict]:
        return self._db.ai_runs.find_one({"_id": event_id})

    def find_run_by_payment(self, payment_id: str) -> Optional[dict]:
        return self._db.ai_runs.find_one({"payment_id": payment_id}, sort=[("created_at", DESCENDING)])

    # decision memory
    def save_decision(self, decision: dict) -> None:
        self._db.decision_memory.insert_one(decision)

    def prior_decisions(self, party_names: list[str], entry_ids: list[str], limit: int = 3) -> list[dict]:
        keys = [_party_key(n) for n in party_names]
        cursor = (self._db.decision_memory.find({"party_key": {"$in": keys}, "watchlist_entry_id": {"$in": entry_ids}}, {"_id": 0})
                  .sort("decided_at", DESCENDING).limit(limit))
        return list(cursor)

    # idempotency
    def is_processed(self, consumer: str, event_id: str) -> bool:
        return self._db.processed_events.find_one({"_id": f"{consumer}:{event_id}"}) is not None

    def mark_processed(self, consumer: str, event_id: str) -> None:
        try:
            self._db.processed_events.insert_one({"_id": f"{consumer}:{event_id}", "at": datetime.now(timezone.utc)})
        except DuplicateKeyError:
            pass     # already marked: that is fine


Repo = MongoRepo | MemoryRepo
party_key = _party_key
