"""
topics.py - Kafka topic names and consumer groups (shared-contracts section 5). Names end in .v1; a breaking change makes .v2.
"""
REQUEST = "ai.review.request.v1"                 # in  : the alert package from the orchestrator          (group ai-svc)
RESULT = "ai.review.result.v1"                   # out : one result per request, always, even on failure
SCREENING_REQUEST = "screening.request.v1"       # out : request_kind RESCREEN
RESCREEN_RESULT = "screening.rescreen.result.v1"  # in : the answer to a RESCREEN request                  (group ai-rescreen-result)
PAYMENT_STATUS = "payment.status.v1"             # in  : final states -> decision memory                   (group ai-decision-memory)
REVIEW_STATUS = "review.status.v1"               # in  : officer decisions -> decision memory               (group ai-decision-memory)
DATASET_ACTIVATED = "sanctions.dataset.activated.v1"   # in : a new list / policy is active              (group ai-dataset)
AUDIT = "audit.events.v1"                        # out : one audit event per meaningful step
DLQ = "ai.dlq.v1"                                # out : messages that can never be processed

GROUP_REQUEST = "ai-svc"
GROUP_RESCREEN = "ai-rescreen-result"
GROUP_MEMORY = "ai-decision-memory"
GROUP_DATASET = "ai-dataset"

SERVICE = "sanction-ai"                           # the name the audit service expects
PRODUCER = f"{SERVICE}@1.0.0"
