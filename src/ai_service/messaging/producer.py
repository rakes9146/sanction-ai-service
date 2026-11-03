"""
producer.py - sending events to Kafka, with the retry rule of the shared contract.

WHAT: `make_envelope` wraps a payload in the standard event envelope; `EventPublisher.publish` sends it.
WHY:  every service sends the same envelope (shared-contracts section 4) and retries the same way (section 6):
      3 attempts in total, 0.5 s apart, the SAME event_id every time, and the header x-attempt = the attempt number.
      If the third attempt fails we raise PublishFailed and the caller does NOT commit the Kafka offset, so the request is not lost.

New to Python? `Producer.produce()` only queues the message. `flush()` waits until Kafka has answered, and the
delivery callback tells us whether it worked.
"""
from __future__ import annotations

import json
from typing import Optional

import structlog
from confluent_kafka import Producer
from tenacity import RetryError, Retrying, stop_after_attempt, wait_fixed

from ai_service.ids import now_iso, uuid7
from ai_service.messaging import topics

log = structlog.get_logger()


class PublishFailed(Exception):
    """Kafka did not accept the message after all attempts."""


def make_envelope(event_type: str, payment_id: str, request_id: Optional[str], causation_id: Optional[str],
                  traceparent: Optional[str], payload: dict, event_id: Optional[str] = None) -> dict:
    return {
        "event_id": event_id or uuid7(),
        "event_type": event_type,
        "event_version": 1,
        "correlation_id": payment_id,                 # = payment_id for the whole life of the payment
        "causation_id": causation_id,                 # the event that triggered this one
        "payment_id": payment_id,
        "screening_request_id": request_id,           # the request id: follow one payment across all services with it
        "producer": topics.PRODUCER,
        "timestamp": now_iso(),
        "schema_version": "1.0.0",
        "trace_context": {"traceparent": traceparent} if traceparent else {},
        "payload": payload,
    }


class EventPublisher:
    def __init__(self, bootstrap_servers: str, attempts: int = 3, backoff_s: float = 0.5) -> None:
        self._producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "acks": "all",                    # wait until the broker has the message safely
            "enable.idempotence": True,       # a retry can never create a duplicate in the topic
            "linger.ms": 0,                   # do not wait to batch: latency matters here
            "compression.type": "lz4",
        })
        self._attempts = attempts
        self._backoff_s = backoff_s

    def publish(self, topic: str, key: str, envelope: dict) -> None:
        """Send one event. Key = payment_id, so all events of one payment stay in order on one partition."""
        value = json.dumps(envelope).encode("utf-8")
        try:
            for attempt in Retrying(stop=stop_after_attempt(self._attempts), wait=wait_fixed(self._backoff_s), reraise=False):
                with attempt:
                    self._send(topic, key, value, envelope, attempt.retry_state.attempt_number)
        except RetryError as e:
            raise PublishFailed(f"could not publish to {topic} after {self._attempts} attempts") from e

    def publish_raw(self, topic: str, key: Optional[str], value: bytes, headers: dict[str, str]) -> None:
        """Used for the DLQ: the ORIGINAL bytes plus the reason, no envelope."""
        self._producer.produce(topic, key=key, value=value, headers=headers)
        self._producer.flush(10)

    def _send(self, topic: str, key: str, value: bytes, envelope: dict, attempt: int) -> None:
        outcome: dict = {}

        def on_delivery(err, msg) -> None:            # called by Kafka's client when the broker has answered
            outcome["error"] = err

        headers = {
            "event_type": envelope["event_type"], "schema_version": envelope["schema_version"], "content-type": "application/json",
            "x-request-id": envelope.get("screening_request_id") or "", "x-payment-id": envelope["payment_id"],
            "traceparent": envelope.get("trace_context", {}).get("traceparent", ""), "x-attempt": str(attempt), "x-synthetic": "true",
        }
        self._producer.produce(topic, key=key, value=value, headers=headers, on_delivery=on_delivery)
        remaining = self._producer.flush(10)
        if remaining or outcome.get("error"):
            log.warning("publish attempt failed", topic=topic, attempt=attempt, error=str(outcome.get("error")))
            raise PublishFailed(str(outcome.get("error") or "timeout"))

    def close(self) -> None:
        self._producer.flush(5)
