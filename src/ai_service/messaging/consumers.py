"""
consumers.py - one small reusable Kafka reading loop.

WHAT: `ConsumerLoop` is a thread that repeats: ask Kafka for up to N messages -> give them to a handler function -> commit.
WHY:  the service has four readers (alerts, re-screen answers, decision memory, dataset changes). They all need the same loop,
      and the rules from shared-contracts section 6 live in this one place:
        * the offset is committed ONLY after the handler finished: if we crash first, Kafka delivers the message again
        * a handler error is tried 3 times, 0.5 s apart; if it still fails the loop STOPS without committing and the service
          reports DOWN on /health (a restart picks up where it left off)
        * a message that can never be processed (not JSON, wrong shape) is not an error: the handler sends it to the DLQ itself

New to Python? `threading.Thread` runs `run()` in the background. `stop_event` is a flag: `stop()` sets it and the loop ends.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import structlog
from confluent_kafka import Consumer, Message

log = structlog.get_logger()


class ConsumerLoop(threading.Thread):
    def __init__(self, name: str, bootstrap_servers: str, group: str, topics: list[str], handler: Callable[[list[Message]], None],
                 batch_size: int = 1, offset_reset: str = "earliest") -> None:
        super().__init__(name=name, daemon=True)
        self.failed = False                      # True once the loop gave up: shown on /health
        self._handler = handler
        self._batch_size = batch_size
        self._stop_event = threading.Event()
        self._config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group,
            "auto.offset.reset": offset_reset,
            "enable.auto.commit": False,         # WE decide when a message counts as done
            "client.id": name,
        }
        self._topics = topics

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        consumer = Consumer(self._config)
        consumer.subscribe(self._topics)
        log.info("consumer started", consumer=self.name, topics=self._topics)
        try:
            while not self._stop_event.is_set():
                messages = [m for m in consumer.consume(num_messages=self._batch_size, timeout=1.0) if m.error() is None]
                if not messages:
                    continue
                if not self._handle_with_retry(messages):
                    return                       # gave up: no commit
                consumer.commit(asynchronous=False)       # only now are these messages "done"
        finally:
            consumer.close()
            log.info("consumer stopped", consumer=self.name)

    def _handle_with_retry(self, messages: list[Message]) -> bool:
        for attempt in (1, 2, 3):
            try:
                self._handler(messages)
                return True
            except Exception as e:               # noqa: BLE001 - any failure counts; the handler logs details
                log.warning("handler failed", consumer=self.name, attempt=attempt, error=type(e).__name__)
                if attempt < 3:
                    time.sleep(0.5)
        self.failed = True
        log.error("consumer stopped WITHOUT committing; restart the service to retry", consumer=self.name)
        return False
