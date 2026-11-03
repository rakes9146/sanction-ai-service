"""
rescreen.py - ask the screening service to match what the AI found, and WAIT for the answer.

WHAT: the graph node `rescreen` calls `RescreenClient.rescreen(...)`. It publishes a screening.request.v1 event with
      request_kind=RESCREEN and then blocks (up to RESCREEN_TIMEOUT_S, default 2 s) until the answer arrives on
      screening.rescreen.result.v1. A second thread (a Kafka consumer, see service.py) calls `deliver(...)` when it arrives.
WHY:  name matching exists only in the Java screening service. The AI must never decide whether a name is on a list.

How the waiting works (a classic pattern, kept as simple as possible):
    1. make a `threading.Event` (a flag that one thread can set and another can wait for) and store it under our request's event_id
    2. publish the request
    3. `event.wait(timeout)`            <- this worker thread sleeps here
    4. the consumer thread receives the answer, sees causation_id == our event_id, stores the answer and calls `event.set()`
    5. we wake up and read the answer. If the timeout passed first, we report ok=False (the graph then raises floor G-07).
The answer is matched by its causation_id (= the event_id of OUR request), so several re-screens can be open at the same time.
"""
from __future__ import annotations

import threading
from typing import Optional

import structlog

from ai_service.graph.state import RescreenOutcome
from ai_service.messaging import topics
from ai_service.messaging.producer import EventPublisher, PublishFailed, make_envelope

log = structlog.get_logger()


class _Waiting:
    """One open re-screen: the flag to wait on and the place where the answer will be put."""

    def __init__(self) -> None:
        self.flag = threading.Event()
        self.payload: Optional[dict] = None


class RescreenClient:
    def __init__(self, publisher: EventPublisher, timeout_s: float, request_timeout_ms: int = 500) -> None:
        self._publisher = publisher
        self._timeout_s = timeout_s
        self._request_timeout_ms = request_timeout_ms
        self._open: dict[str, _Waiting] = {}
        self._lock = threading.Lock()          # protects _open: several worker threads and the consumer thread use it

    # ---- called by the graph (worker thread) --------------------------------------------------------------------------
    def rescreen(self, payment_id: str, request_id: str, traceparent: Optional[str], items: list[dict]) -> RescreenOutcome:
        payload = {"request_kind": "RESCREEN", "reply_topic": topics.RESCREEN_RESULT, "parsed_message": None,
                   "timeout_ms": self._request_timeout_ms, "items": items}
        envelope = make_envelope("ScreeningRequested", payment_id, request_id, None, traceparent, payload)
        event_id = envelope["event_id"]
        waiting = _Waiting()
        with self._lock:
            self._open[event_id] = waiting        # register BEFORE publishing: the answer could be very fast
        try:
            self._publisher.publish(topics.SCREENING_REQUEST, payment_id, envelope)
        except PublishFailed:
            self._forget(event_id)
            return RescreenOutcome(ok=False, hits=[], floor_hints=[], reason="could not send the re-screen request")
        answered = waiting.flag.wait(self._timeout_s)        # <- this thread sleeps until deliver() sets the flag, or time runs out
        self._forget(event_id)
        if not answered or waiting.payload is None:
            log.warning("re-screen timed out", request_id=request_id, seconds=self._timeout_s)
            return RescreenOutcome(ok=False, hits=[], floor_hints=[], reason="timeout")
        return self._outcome(waiting.payload)

    # ---- called by the consumer thread --------------------------------------------------------------------------------
    def deliver(self, causation_id: Optional[str], payload: dict) -> bool:
        """An answer arrived. True if somebody was waiting for it, False if nobody (late or foreign answer: ignored)."""
        with self._lock:
            waiting = self._open.get(causation_id or "")
        if waiting is None:
            return False
        waiting.payload = payload
        waiting.flag.set()                        # wake the worker thread
        return True

    # ---- helpers --------------------------------------------------------------------------------------------------------
    def _forget(self, event_id: str) -> None:
        with self._lock:
            self._open.pop(event_id, None)

    @staticmethod
    def _outcome(payload: dict) -> RescreenOutcome:
        """Turn the screening service's answer (sample 07) into what the graph needs. Anything but COMPLETED means 'could not verify'."""
        if payload.get("status") != "COMPLETED" or payload.get("incomplete"):
            return RescreenOutcome(ok=False, hits=[], floor_hints=[], reason=f"screening status {payload.get('status')}")
        hits = []
        for item in payload.get("item_results", []):
            for h in item.get("hits", []):
                hits.append({**h, "item_id": item.get("item_id")})
        return RescreenOutcome(ok=True, hits=hits, floor_hints=list(payload.get("floor_hints", [])))
