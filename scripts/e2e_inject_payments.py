"""
e2e_inject_payments.py - end-to-end check of ALL services together (orchestrator + sanction-check + this AI service).

It publishes nine mock payments to payments.inbound.v1, waits, then reads payment.status / screening / review topics and prints,
for every payment, the path it took (RECEIVED -> SCREENING -> AI_REVIEW -> RELEASED | REVIEW_PENDING_L3 ...).

    uv run --with confluent-kafka python scripts/e2e_inject_payments.py [seconds]          # inject and report (default 30 s)
    uv run --with confluent-kafka python scripts/e2e_inject_payments.py 12 read            # only report what is already on Kafka

KAFKA_BOOTSTRAP (default localhost:9092) says where Kafka is. Needs the three services running and the topics created.
Expected: neft_clean -> RELEASED without AI; "Mohd Abdul Hafiz" (neft_level2_transliteration) -> AI_REVIEW -> RELEASED;
everything else -> REVIEW_PENDING_L3 with the floors in the report.
"""
import os
from pathlib import Path

import json, sys, time, uuid
from confluent_kafka import Consumer, Producer

BOOT = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
SAMPLES = str(Path(__file__).resolve().parents[2] / "payment-mock-service" / "contracts" / "samples" / "inbound") + "/"
CASES = ["neft_clean", "rtgs_level1_listed_entity", "mt103_level3_tehran_street", "neft_level1_restricted_ifsc",
         "pacs008_level1_banned_intermediary", "neft_level2_transliteration", "mt103_level3_concealment",
         "pacs008_level3_injection_near_match", "rtgs_level3_location_word"]

def ulid_like():
    return uuid.uuid4().hex[:26].upper().replace("I", "1").replace("L", "2").replace("O", "0").replace("U", "V")

prod = Producer({"bootstrap.servers": BOOT})
sent = {}
READ_ONLY = len(sys.argv) > 2 and sys.argv[2] == "read"
for name in ([] if READ_ONLY else CASES):
    e = json.load(open(SAMPLES + name + ".json"))
    pay = "pay_" + ulid_like()
    e["event_id"] = "01a14afe-" + uuid.uuid4().hex[:4] + "-79e1-8ebc-" + uuid.uuid4().hex[:12]
    e["payment_id"] = pay
    e["correlation_id"] = pay
    e["payload"]["txn_ref"] = "E2E" + uuid.uuid4().hex[:12]
    prod.produce("payments.inbound.v1", key=pay, value=json.dumps(e))
    sent[pay] = name
prod.flush()
print("published", len(sent), "payments")

c = Consumer({"bootstrap.servers": BOOT, "group.id": "e2e-" + uuid.uuid4().hex[:6], "auto.offset.reset": "earliest"})
topics = ["payment.status.v1", "screening.request.v1", "screening.result.v1", "ai.review.request.v1",
          "review.requested.v1", "audit.events.v1", "orchestrator.dlq.v1", "screening.dlq.v1"]
c.subscribe(topics)
events = {}
deadline = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 25)
while time.time() < deadline:
    m = c.poll(0.5)
    if m is None or m.error():
        continue
    try:
        j = json.loads(m.value())
    except Exception:
        print("non-json on", m.topic())
        continue
    pay = j.get("payment_id")
    if pay and pay.startswith("pay_"):
        events.setdefault(pay, []).append((m.topic(), j))
c.close()
if READ_ONLY:
    sent = {p: p[-6:] for p in events}

for pay, name in sent.items():
    ev = events[pay]
    states = [j["payload"]["state"] for t, j in ev if t == "payment.status.v1"]
    res = [j["payload"] for t, j in ev if t == "screening.result.v1"]
    hits = [(h["rule_id"], h["watchlist_entry_id"]) for r in res for h in r["hits"]]
    req = [j for t, j in ev if t == "screening.request.v1"]
    rid = req[0]["screening_request_id"] if req else None
    audits = sorted({j["payload"]["service"] + ":" + j["payload"]["action"] for t, j in ev if t == "audit.events.v1"})
    final = [j["payload"] for t, j in ev if t == "payment.status.v1"][-1:] or [{}]
    review = [j["payload"] for t, j in ev if t == "review.requested.v1"]
    print("\n== %s  (%s)  request_id=%s" % (name, pay, rid))
    print("   states:", " -> ".join(states))
    print("   screening hits:", hits, "| floor_hints:", [r["floor_hints"] for r in res])
    print("   AI requested:", any(t == "ai.review.request.v1" for t, _ in ev),
          "| review case:", [(r["tier"], r["floors"]) for r in review])
    print("   final:", {k: final[0].get(k) for k in ("state", "disposition", "reason_code", "screening_outcome", "payment_version")})
    print("   audit:", audits)
