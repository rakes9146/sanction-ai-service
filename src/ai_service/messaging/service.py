"""
service.py - wires everything together and handles each kind of Kafka message.

Read this file top to bottom to see the whole life of one alert:

    Kafka ai.review.request.v1
      -> AiService._on_requests        decode (bad message -> DLQ), hand each alert to a worker thread
      -> AiService._process_alert      (in the worker)
            1. already answered this request?  re-send the stored answer, do not run the graph again
            2. run the graph                   graph/builder.py: run_alert(...)
            3. save the run in MongoDB         ai_runs  (before publishing, so an answer can be re-sent later)
            4. publish ai.review.result.v1     ALWAYS one result per request, also when it failed
            5. publish the audit events        best effort
            6. mark the request processed
      -> the reader thread commits the Kafka offset (consumers.py)

Other readers: re-screen answers (wake the waiting worker), payment status (decision memory), dataset changes (refresh the policy store).
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import structlog
from confluent_kafka import Message

from ai_service.config import Rules, Settings
from ai_service.graph.builder import build_graph, run_alert
from ai_service.graph.state import Deps
from ai_service.ids import now_iso, result_event_id, ulid
from ai_service.llm import make_llm
from ai_service.tracing import flush_traces, setup_tracing
from ai_service.messaging import topics
from ai_service.messaging.consumers import ConsumerLoop
from ai_service.messaging.producer import EventPublisher, PublishFailed, make_envelope
from ai_service.messaging.rescreen import RescreenClient
from ai_service.rag.embedder import make_embedder
from ai_service.rag.ingest import ingest
from ai_service.rag.retriever import PolicyRetriever
from ai_service.rag.store import PolicyStore
from ai_service.repo import MemoryRepo, MongoRepo, party_key

log = structlog.get_logger()
_FAILURE_WORDS = ("FAILED", "INVALID", "GATE_FAILED", "EVIDENCE_DROPPED")


class AiService:
    def __init__(self, settings: Settings, repo: Optional[MongoRepo | MemoryRepo] = None) -> None:
        settings.check_openai_ready()
        setup_tracing(settings)                 # LangSmith on/off from config/application.yml (before the first alert)
        self.settings = settings
        self.rules = Rules.load(settings.config_dir)
        self.repo = repo or MongoRepo(settings)
        self.publisher = EventPublisher(settings.kafka_bootstrap)

        # RAG: make sure the policy clauses are in the vector store (safe to repeat; see rag/ingest.py)
        self.embedder = make_embedder(settings)
        self.store = PolicyStore(path=None if settings.qdrant_url else settings.qdrant_path, url=settings.qdrant_url or None)
        log.info("policy ingest", report=ingest(self.store, self.embedder).summary())
        self.retriever = PolicyRetriever(self.store, self.embedder)

        self.rescreen = RescreenClient(self.publisher, settings.rescreen_timeout_s)
        self.deps = Deps(settings, self.rules, make_llm(settings), self.retriever, self.rescreen, self.repo)
        self.graph = build_graph(self.deps)
        self.pool = ThreadPoolExecutor(max_workers=settings.ai_max_concurrency, thread_name_prefix="alert")

        n = settings.ai_max_concurrency
        self.loops = [
            ConsumerLoop("ai-requests", settings.kafka_bootstrap, topics.GROUP_REQUEST, [topics.REQUEST], self._on_requests, batch_size=n),
            ConsumerLoop("ai-rescreen", settings.kafka_bootstrap, topics.GROUP_RESCREEN, [topics.RESCREEN_RESULT], self._on_rescreen_results,
                         batch_size=20, offset_reset="latest"),
            ConsumerLoop("ai-memory", settings.kafka_bootstrap, topics.GROUP_MEMORY, [topics.PAYMENT_STATUS, topics.REVIEW_STATUS],
                         self._on_status_events, batch_size=20),
            ConsumerLoop("ai-dataset", settings.kafka_bootstrap, topics.GROUP_DATASET, [topics.DATASET_ACTIVATED], self._on_dataset),
        ]

    # ---- lifecycle ---------------------------------------------------------------------------------------------------
    def start(self) -> None:
        for loop in self.loops:
            loop.start()

    def stop(self) -> None:
        for loop in self.loops:
            loop.stop()
        for loop in self.loops:
            loop.join(timeout=5)
        self.pool.shutdown(wait=True, cancel_futures=False)
        self.publisher.close()
        flush_traces(self.settings)                 # last step: the alerts above have finished, now ship their traces

    def health(self) -> dict:
        failed = [l.name for l in self.loops if l.failed]
        return {"status": "DOWN" if failed else "UP", "stopped_consumers": failed,
                "llm_provider": self.settings.llm_provider, "ai_enabled": self.settings.ai_enabled}

    # ---- 1. alerts ----------------------------------------------------------------------------------------------------
    def _on_requests(self, messages: list[Message]) -> None:
        """Decode the batch, run the alerts in parallel (one worker thread each), wait for all of them."""
        futures = []
        for message in messages:
            envelope = self._decode(message)
            if envelope is not None:
                futures.append(self.pool.submit(self._process_alert, envelope, message.partition(), message.offset()))
        for future in futures:
            future.result()                     # re-raises a worker error, so the loop does not commit

    def _process_alert(self, envelope: dict, partition: int, offset: int) -> None:
        event_id = envelope["event_id"]
        structlog.contextvars.bind_contextvars(requestId=envelope.get("screening_request_id"), paymentId=envelope.get("payment_id"),
                                               eventId=event_id)
        try:
            log.info("alert received", partition=partition, offset=offset, topic=topics.REQUEST)
            previous = self.repo.find_run(event_id)
            if previous is not None:            # the same request again (redelivery, or the orchestrator re-sent it): do not pay for the LLM twice
                log.info("duplicate request: re-sending the stored result")
                self.publisher.publish(topics.RESULT, envelope["payment_id"], previous["result_event"])
                self.repo.mark_processed(topics.GROUP_REQUEST, event_id)
                return

            state = run_alert(self.graph, self.deps, envelope)
            result = state["result"]
            result_envelope = make_envelope("AiReviewCompleted", envelope["payment_id"], envelope.get("screening_request_id"), event_id,
                                            envelope.get("trace_context", {}).get("traceparent"), result,
                                            event_id=result_event_id(event_id))
            self.repo.save_run(self._run_document(envelope, state, result_envelope))        # save BEFORE publishing
            self.publisher.publish(topics.RESULT, envelope["payment_id"], result_envelope)
            self._publish_audit(envelope, state)
            self.repo.mark_processed(topics.GROUP_REQUEST, event_id)
            log.info("result published", status=result["status"], route=result["proposed_route"], score=result["alert_score"],
                     floors=result["floors_applied"], level=result["ai_level_reached"], timings_ms=state.get("timings", {}))
        finally:
            structlog.contextvars.clear_contextvars()

    @staticmethod
    def _run_document(envelope: dict, state: dict, result_envelope: dict) -> dict:
        """What is stored in ai_runs: enough to audit, replay, re-send the answer and build decision memory. No message text."""
        request = state.get("request")
        hits = [{"hit_id": h.hit_id, "rule_id": h.rule_id, "original_value": h.original_value, "watchlist_entry_id": h.watchlist_entry_id}
                for h in (request.alert.hits if request else [])]
        return {"_id": envelope["event_id"], "payment_id": envelope.get("payment_id"), "request_id": envelope.get("screening_request_id"),
                "created_at": datetime.now(timezone.utc), "hits": hits, "status": state["status"], "route": state["proposed_route"],
                "alert_score": state["result"]["alert_score"], "timings": state.get("timings", {}), "usage": state.get("usage", []),
                "errors": state.get("errors", []), "versions": state["result"]["versions"], "result_event": result_envelope}

    def _publish_audit(self, envelope: dict, state: dict) -> None:
        """One audit event per entry in state['audit']. Best effort: a failure here must never lose the result."""
        for entry in state.get("audit", []):
            action = entry["action"]
            details = entry["details"]
            payload = {"audit_event_id": "aud_" + ulid(), "service": topics.SERVICE, "action": action, "stage": "AI",
                       "outcome": "FAILURE" if any(w in action for w in _FAILURE_WORDS) else "SUCCESS",
                       "actor": {"type": "SYSTEM", "id": topics.SERVICE}, "summary": f"{action} for {envelope.get('payment_id')}"[:200],
                       "payload_hash": "sha256:" + hashlib.sha256(json.dumps(details, sort_keys=True, default=str).encode()).hexdigest(),
                       "details": details, "occurred_at": now_iso()}
            audit = make_envelope("AuditEventRecorded", envelope["payment_id"], envelope.get("screening_request_id"), envelope["event_id"],
                                  envelope.get("trace_context", {}).get("traceparent"), payload)
            try:
                self.publisher.publish(topics.AUDIT, envelope["payment_id"], audit)
            except PublishFailed:
                log.warning("audit event not published", action=action)

    # ---- 2. re-screen answers -------------------------------------------------------------------------------------------
    def _on_rescreen_results(self, messages: list[Message]) -> None:
        for message in messages:
            envelope = self._decode(message)
            if envelope is not None:
                self.rescreen.deliver(envelope.get("causation_id"), envelope.get("payload", {}))

    # ---- 3. decision memory ------------------------------------------------------------------------------------------------
    def _on_status_events(self, messages: list[Message]) -> None:
        """When a payment reaches a final state, remember the decision against each party + list entry of its alert."""
        for message in messages:
            envelope = self._decode(message)
            if envelope is None or message.topic() != topics.PAYMENT_STATUS:
                continue
            payload = envelope.get("payload", {})
            if not payload.get("is_terminal") or payload.get("disposition") not in ("RELEASE", "REJECT", "BLOCK"):
                continue
            if self.repo.is_processed(topics.GROUP_MEMORY, envelope["event_id"]):
                continue
            run = self.repo.find_run_by_payment(envelope["payment_id"])
            for hit in (run or {}).get("hits", []):
                if hit.get("watchlist_entry_id") and hit.get("original_value"):
                    self.repo.save_decision({"party_key": party_key(hit["original_value"]), "watchlist_entry_id": hit["watchlist_entry_id"],
                                             "decision": payload["disposition"], "decided_by": "OFFICER" if payload.get("applied_decision_id") else "AI",
                                             "decided_at": datetime.now(timezone.utc), "request_id": envelope.get("screening_request_id"),
                                             "payment_id": envelope["payment_id"]})
            self.repo.mark_processed(topics.GROUP_MEMORY, envelope["event_id"])

    # ---- 4. dataset / policy changed ---------------------------------------------------------------------------------------
    def _on_dataset(self, messages: list[Message]) -> None:
        log.info("dataset activated: refreshing the policy store", report=ingest(self.store, self.embedder).summary())

    # ---- shared ----------------------------------------------------------------------------------------------------------------
    def _decode(self, message: Message) -> Optional[dict]:
        """Bytes -> envelope. A message that cannot be understood goes to the DLQ (with the original bytes and the reason) and is skipped."""
        try:
            envelope = json.loads(message.value())
            if not isinstance(envelope, dict) or not envelope.get("event_id") or not isinstance(envelope.get("payload"), dict):
                raise ValueError("missing event_id or payload")
            if message.topic() in (topics.REQUEST, topics.RESCREEN_RESULT, topics.PAYMENT_STATUS) and not envelope.get("payment_id"):
                raise ValueError("missing payment_id")
            return envelope
        except Exception as e:                  # noqa: BLE001 - anything wrong with the message means "poison"
            reason = f"{type(e).__name__}: {e}"[:200]
            log.warning("poison message sent to the DLQ", topic=message.topic(), partition=message.partition(), offset=message.offset(), reason=reason)
            key = message.key().decode("utf-8", "replace") if message.key() else None
            self.publisher.publish_raw(topics.DLQ, key, message.value() or b"", {
                "dlq-reason": reason, "dlq-source-topic": message.topic(), "dlq-source-partition": str(message.partition()),
                "dlq-source-offset": str(message.offset())})
            return None
