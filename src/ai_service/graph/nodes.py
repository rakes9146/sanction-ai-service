"""
nodes.py - the TEN STEPS of the graph, in the order they run.

How to read this file:
    * Each step is a method of `Nodes` that takes the notebook (`state`) and returns a small dict with only the keys it changes.
    * Steps marked [CODE] are ordinary deterministic code. Only `l1_call` and `l2_call` ever talk to the language model.
    * "RC-nn" / "PA-nn" / "G-nn" are the ids of the catalogue's re-checks, pre-AI checks and safety floors.

    prepare ──► plan_l1 ──► l1_call ──► verify_l1 ──► rescreen ──► score ──┬──► emit            (cleared by score alone)
     [CODE]     [CODE]       [AI]        [CODE]        [CODE]      [CODE]   └──► l2_retrieve ──► l2_call ──► final_gate ──► emit
                                                                                   [CODE+RAG]       [AI]        [CODE]      [CODE]

    If any step fails it sets failed=True and the graph jumps to `emit`, which publishes a FAILED result (the orchestrator then
    sends the payment to a human). Every path ends in `emit`.

Core rule: the AI labels and explains, CODE verifies, scores and decides.
"""
from __future__ import annotations

import re

import structlog

from ai_service.checks.attribute_compare import UNKNOWN, compare_attributes, compare_value, facts_for_hit
from ai_service.checks.evidence import evidence_found
from ai_service.checks.floors import highest_floor_level, union
from ai_service.checks.ownership import listed_owners
from ai_service.checks.qa_sampling import is_qa_sample
from ai_service.checks.scoring import (aligned_name_similarity, band_for, effective_band, location_score, name_hit_score)
from ai_service.graph.state import AlertState, Deps
from ai_service.llm.base import L1Request, L2Request
from ai_service.models import (L1_FIELD_BY_CHECK, AiRequestPayload, L1Output, L2Output, LLMSchemaError)

log = structlog.get_logger()

L1_PROMPT_VERSION = "l1-read-v1"
L2_PROMPT_VERSION = "l2-reason-v1"
TEXT_KINDS = ("NAME", "ADDRESS", "FREE_TEXT")      # the field kinds that are shown to the model


def _add(audit: list[dict], action: str, **details) -> list[dict]:
    """A copy of the audit list with one more entry. (Nodes return the whole new list; there are no special merge rules.)"""
    return [*audit, {"action": action, "details": details}]


class Nodes:
    def __init__(self, deps: Deps) -> None:
        self.d = deps

    # =================================================================================================================
    # 1. prepare [CODE]
    #    WHAT: check the request, honour the kill switch, collect the payment text, run PA-02 (attribute compare).
    #    WHY:  everything later needs the same tidy inputs; and we must stop BEFORE any paid LLM call if the AI is switched off.
    # =================================================================================================================
    def prepare(self, state: AlertState) -> dict:
        payload: AiRequestPayload = state["request"]
        settings, rules = self.d.settings, self.d.rules
        audit = _add(state.get("audit", []), "AI_REQUEST_RECEIVED", hits=len(payload.alert.hits), format=payload.format)

        # kill switch: env AI_ENABLED=false, a runtime flag in MongoDB, or the flag inside the request
        if not settings.ai_enabled or payload.ai_kill_switch or self.d.repo.get_flag("ai_kill_switch"):
            return {"failed": True, "errors": ["AI_KILL_SWITCH"], "audit": _add(audit, "AI_FAILED", reason="AI_KILL_SWITCH")}

        parsed = payload.untrusted_message.get("parsed_message") or {}
        fields = parsed.get("screenable_fields") or []
        if not payload.alert.hits or not fields:
            return {"failed": True, "errors": ["INVALID_REQUEST"], "audit": _add(audit, "AI_FAILED", reason="INVALID_REQUEST")}

        # the text the model may read (names, addresses, free text) and the text quotes are checked against (everything)
        texts = [{"field_id": f["field_id"], "path": f["path"], "kind": f["kind"], "value": f["value"]}
                 for f in fields if f.get("kind") in TEXT_KINDS and f.get("value")]
        all_texts = [str(f["value"]) for f in fields if f.get("value")] + [str(t["value"]) for t in parsed.get("free_text", []) if t.get("value")]

        # floors the orchestrator already raised; an injection found by its regex scan (PA-03) always means G-05
        floors = list(payload.alert.floors_so_far)
        security_event = bool(payload.alert.pre_ai_checks.get("PA-03", {}).get("injection_found"))
        if security_event:
            floors = union(floors, ["G-05"])

        # PA-02: for every NAME hit compare DOB / nationality / city / identifiers with the listed person (code, not AI)
        name_rules = set(rules.scoring["name_rules"])
        customer = payload.customer.model_dump()
        table = []
        for hit in payload.alert.hits:
            if hit.rule_id in name_rules:
                row = compare_attributes(hit.list_entry or {}, facts_for_hit(hit.field_id, parsed, customer))
                table.append({"hit_id": hit.hit_id, **row})
        audit = _add(audit, "AI_PREPARED", name_hits=len(table), floors=floors, security_event=security_event)
        return {"texts": texts, "all_texts": all_texts, "floors": floors, "security_event": security_event,
                "attribute_table": table, "audit": audit, "errors": [], "usage": [], "timings": {}, "failed": False}

    # =================================================================================================================
    # 2. plan_l1 [CODE]
    #    WHAT: from the kind of hits, switch on only the L1 questions that can add something (config/l1_plan.yml).
    #    WHY:  fewer questions = shorter prompt = cheaper, faster, and a model that stays on the point.
    # =================================================================================================================
    def plan_l1(self, state: AlertState) -> dict:
        plan_cfg = self.d.rules.l1_plan
        checks: set[str] = set()
        for hit in state["request"].alert.hits:
            checks.update(plan_cfg["hit_rules"].get(hit.rule_id, []))
        if any(t["kind"] == "FREE_TEXT" for t in state["texts"]):
            checks.update(plan_cfg["always_if_free_text"])          # L1-09 manipulation check on every alert with free text
        return {"l1_plan": sorted(checks)}

    # =================================================================================================================
    # 3. l1_call [AI #1]
    #    WHAT: ONE call to the model with the switched-on questions. The answer must fit the L1Output shape.
    #    WHY:  this is the "reading" an L1 analyst does. If the shape is wrong we try once more, then give up (floor G-06).
    # =================================================================================================================
    def l1_call(self, state: AlertState) -> dict:
        plan = state.get("l1_plan", [])
        if not plan:                                                # nothing to ask: code alone scores these hits
            return {"audit": _add(state.get("audit", []), "AI_L1_COMPLETED", checks=[], skipped=True)}
        request = L1Request(checks=plan, hits=self._compact_hits(state), texts=state["texts"])
        try:
            output, usage = self.d.llm.read(request)
        except LLMSchemaError:
            log.warning("l1 answer had the wrong shape, trying once more")
            output, usage = self.d.llm.read(request)               # a second LLMSchemaError goes up to the guard -> AI_OUTPUT_INVALID
        return {"l1_raw": output, "usage": [*state.get("usage", []), usage.__dict__],
                "audit": _add(state.get("audit", []), "AI_L1_COMPLETED", checks=plan, tokens=usage.prompt_tokens + usage.completion_tokens)}

    # =================================================================================================================
    # 4. verify_l1 [CODE]  RC-01 + RC-02
    #    RC-01 (shape) was already enforced by Pydantic when the answer was parsed.
    #    RC-02 (evidence is real): every quote must be an exact piece of the payment. An answer with a made-up quote is DROPPED
    #    (treated as UNKNOWN). The AI may not rely on facts that are not in the message.
    # =================================================================================================================
    def verify_l1(self, state: AlertState) -> dict:
        raw: L1Output | None = state.get("l1_raw")
        if raw is None:
            return {"l1": None, "dropped_evidence": []}
        texts = state["all_texts"]
        dropped: list[dict] = []

        def keep(item, check: str) -> bool:
            if getattr(item, "label", None) in ("NONE", "UNKNOWN"):    # "nothing found" needs no quote
                return True
            ok = evidence_found(getattr(item, "evidence", ""), texts)
            if not ok:
                dropped.append({"check": check, "evidence": getattr(item, "evidence", "")})
            return ok

        verified = L1Output(
            token_roles=[i for i in raw.token_roles if keep(i, "L1-01")],
            party_types=[i for i in raw.party_types if keep(i, "L1-02")],
            name_parts=[i for i in raw.name_parts if keep(i, "L1-03")],
            entities=[i for i in raw.entities if keep(i, "L1-04")],
            places=[i for i in raw.places if keep(i, "L1-05")],
            goods=raw.goods if raw.goods is None or keep(raw.goods, "L1-06") else None,
            concealment=raw.concealment if raw.concealment is None or keep(raw.concealment, "L1-07") else None,
            attributes=[i for i in raw.attributes if keep(i, "L1-08")],
            manipulation=raw.manipulation,
        )
        audit = state.get("audit", [])
        if dropped:
            audit = _add(audit, "EVIDENCE_DROPPED", count=len(dropped), checks=sorted({d["check"] for d in dropped}))
        return {"l1": verified, "dropped_evidence": dropped, "audit": audit}

    # =================================================================================================================
    # 5. rescreen [CODE, over Kafka]  RC-03, RC-04, RC-05
    #    WHAT: anything the AI EXTRACTED (names in the text, places, ids) is sent back to the screening service to be matched.
    #    WHY:  matching against lists is done in exactly one place (the Java screening service). The AI finds things; it never
    #          decides whether they are on a list.
    #    If the screening service does not answer in time we cannot verify: floor G-07 (a human decides) and we carry on.
    # =================================================================================================================
    def rescreen(self, state: AlertState) -> dict:
        l1 = state.get("l1")
        items: list[dict] = []
        if l1 is not None:
            for e in l1.entities:                                    # L1-04 -> RC-03
                if e.type in ("PERSON", "ORGANISATION"):
                    items.append({"item_id": f"r{len(items) + 1}", "kind": "NAME", "value": e.name, "source_check": "L1-04"})
            for p in l1.places:                                      # L1-05 -> RC-04
                items.append({"item_id": f"r{len(items) + 1}", "kind": "PLACE", "value": p.place, "source_check": "L1-05"})
            for a in l1.attributes:                                  # L1-08 -> RC-05
                if a.type in ("IMO", "PASSPORT", "NATIONAL_ID", "LEI"):
                    items.append({"item_id": f"r{len(items) + 1}", "kind": "IDENTIFIER", "value": a.value, "scheme": a.type,
                                  "source_check": "L1-08"})
        if not items:
            return {"rescreen_status": "NOT_NEEDED", "rescreen_items": [], "rescreen_hits": []}

        outcome = self.d.rescreener.rescreen(state["payment_id"], state["request_id"], state["envelope"].get("trace_context", {}).get("traceparent"), items)
        floors = state["floors"]
        if not outcome.ok:
            floors = union(floors, [self.d.rules.floors["from_ai"]["rescreen_failed"]])      # G-07: could not verify
        else:
            floors = union(floors, outcome.floor_hints)
        hits = [{**h, "item_kind": next((i["kind"] for i in items if i["item_id"] == h.get("item_id")), None)} for h in outcome.hits]

        # RC-05: a DOB found in the text can settle a DOB comparison that was UNKNOWN
        table = [dict(row) for row in state.get("attribute_table", [])]
        for a in (l1.attributes if l1 else []):
            if a.type == "DOB":
                for row, hit in zip(table, [h for h in state["request"].alert.hits if h.rule_id in set(self.d.rules.scoring["name_rules"])]):
                    if row["dob"] == UNKNOWN:
                        row["dob"] = compare_value((hit.list_entry or {}).get("dob"), a.value)
        return {"rescreen_status": "COMPLETED" if outcome.ok else "TIMEOUT", "rescreen_items": items, "rescreen_hits": hits,
                "floors": floors, "attribute_table": table,
                "audit": _add(state.get("audit", []), "RESCREEN_RESULT", ok=outcome.ok, items=len(items), new_hits=len(hits))}

    # =================================================================================================================
    # 6. score [CODE]  RC-06, RC-07, RC-08, RC-09
    #    WHAT: give every hit a score, take the highest as the alert score, pick a band, add the floors the AI's findings imply.
    #    WHY:  the SCORE decides how much human attention a payment needs. It must be reproducible, so it is arithmetic (see
    #          checks/scoring.py and config/scoring.yml), never the model's opinion.
    # =================================================================================================================
    def score(self, state: AlertState) -> dict:
        cfg = self.d.rules.scoring
        floors_cfg = self.d.rules.floors
        l1 = state.get("l1") or L1Output()
        payload: AiRequestPayload = state["request"]
        table = {r["hit_id"]: r for r in state.get("attribute_table", [])}
        scores: list[dict] = []
        notes: list[str] = []

        for hit in payload.alert.hits:
            kind, value, note = self._score_hit(hit.model_dump(), table.get(hit.hit_id), l1, cfg, state)
            scores.append({"hit_id": hit.hit_id, "rule_id": hit.rule_id, "score": round(value, 2), "how": kind})
            if note:
                notes.append(note)
        for hit in state.get("rescreen_hits", []):                    # new hits from re-screening are scored like any other
            kind, value, note = self._score_hit(hit, {"dob": UNKNOWN, "nationality": UNKNOWN, "city": UNKNOWN, "identifiers": UNKNOWN}, l1, cfg, state)
            scores.append({"hit_id": hit.get("item_id"), "rule_id": hit.get("rule_id"), "score": round(value, 2), "how": "rescreen " + kind})

        alert_score = round(max((s["score"] for s in scores), default=0.0), 2)

        # RC-08: risk flags found by the AI turn into floors
        floors = list(state["floors"])
        security_event = state.get("security_event", False)
        from_ai = self.d.rules.floors["from_ai"]
        if l1.concealment and l1.concealment.label not in ("NONE", "UNKNOWN"):
            floors = union(floors, [from_ai["concealment_not_none"]])
        if l1.goods and l1.goods.dual_use_candidate:
            floors = union(floors, [from_ai["dual_use"]])
        if l1.manipulation:
            floors = union(floors, [from_ai["manipulation"]])
            security_event = True
        for place in l1.places:
            if place.sanctioned_region:
                floors = union(floors, [from_ai["sanctioned_place"]])
        if security_event:
            floors = union(floors, ["G-05"])

        raw_band = band_for(alert_score, cfg)
        eff_band = effective_band(raw_band, highest_floor_level(floors, floors_cfg), floors_cfg)   # a floor can only RAISE the band
        rechecks = {"RC-01": "PASS", "RC-02": f"{len(state.get('dropped_evidence', []))} dropped",
                    "RC-07": "; ".join(notes) or "no place hits", "RC-08": f"floors {floors}",
                    "RC-09": f"alert score {alert_score} -> raw band {raw_band}, effective band {eff_band}"}
        return {"hit_scores": scores, "alert_score": alert_score, "raw_band": raw_band, "effective_band": eff_band, "floors": floors,
                "security_event": security_event, "code_rechecks": rechecks,
                "audit": _add(state.get("audit", []), "AI_SCORED", alert_score=alert_score, raw_band=raw_band, effective_band=eff_band, floors=floors)}

    def _score_hit(self, hit: dict, row: dict | None, l1: L1Output, cfg: dict, state: AlertState) -> tuple[str, float, str]:
        """(how it was scored, score, note) for ONE hit. The rule id decides which kind of score applies."""
        rule = hit.get("rule_id", "")
        if rule in cfg["name_rules"]:
            similarity = hit.get("similarity") or 1.0
            if rule in ("S2-R01", "S2-R02", "S1-R01"):                # RC-06: re-score on aligned name parts when the AI found them
                parts = next((p for p in l1.name_parts if p.subject == hit.get("original_value")), None)
                listed = (hit.get("list_entry") or {}).get("name", "")
                aligned = aligned_name_similarity(parts.given, parts.family, listed) if parts else None
                if aligned is not None and rule != "S1-R01":
                    similarity = aligned
            return "name", name_hit_score(similarity, row or {}, cfg), ""
        if rule in cfg["full_score_rules"]:
            return "full", 1.0, ""
        if rule in cfg["location_rules"]:                             # RC-07
            token = (hit.get("matched_text") or "").upper()
            place = next((p for p in l1.places if token and token in p.place.upper()), None)
            label = next((t.label for t in l1.token_roles if t.token.upper() == token), None)
            role = "STREET_NAME" if label == "STREET_NAME" else (place.role if place else "UNKNOWN")
            region = place.sanctioned_region if place else None
            if any(h.get("rule_id") in ("S1-R04", "S2-R04") and h.get("item_kind") == "PLACE" and (h.get("floor_hint") == "G-01") for h in state.get("rescreen_hits", [])):
                region = True                                          # re-screen says the place is a sanctioned country/region
            value = location_score(region, role, cfg)
            return "location", value, f"{token or 'place'}: role {role}, sanctioned_region {region} -> {value}"
        return "other", float(cfg["other_hit_score"]), ""

    # =================================================================================================================
    # 7. l2_retrieve [CODE + RAG]
    #    WHAT: pick the (at most) 5 policy clauses to show AI L2, and look up earlier decisions on the same party.
    #    WHY:  the AI does not know our policy; we put the relevant clauses in front of it (retrieval, see rag/retriever.py).
    # =================================================================================================================
    def l2_retrieve(self, state: AlertState) -> dict:
        payload: AiRequestPayload = state["request"]
        l1 = state.get("l1") or L1Output()
        tags = {h.rule_id for h in payload.alert.hits} | {h.get("rule_id") for h in state.get("rescreen_hits", [])}
        if any(t.label == "STREET_NAME" for t in l1.token_roles):
            tags.add("STREET_NAME")
        if state.get("security_event"):
            tags.add("INJECTION")
        if l1.goods and l1.goods.dual_use_candidate:
            tags.add("DUAL_USE")
        if l1.concealment and l1.concealment.label not in ("NONE", "UNKNOWN"):
            tags.add("CONCEALMENT")
        tags.discard(None)
        query = (f"payment held for review. hit types {' '.join(sorted(tags))}. recommend the next step and the wording of any "
                 f"request for information")
        clauses = [c.as_dict() for c in self.d.retriever.retrieve(tags, payload.format, query)]

        names = [h.original_value for h in payload.alert.hits if h.original_value]
        entries = [h.watchlist_entry_id for h in payload.alert.hits if h.watchlist_entry_id]
        prior = self.d.repo.prior_decisions(names, entries, limit=3)
        return {"clauses": clauses, "prior_decisions": prior,
                "audit": _add(state.get("audit", []), "AI_L2_RETRIEVED", clause_ids=[c["clause_id"] for c in clauses], prior=len(prior))}

    # =================================================================================================================
    # 8. l2_call [AI #2]
    #    WHAT: give the model the VERIFIED facts + the 5 clauses; it cites clause ids, explains, recommends, drafts an RFI.
    #    WHY:  this is the "thinking" an L2 analyst does. It cannot change the score or the band, and "recommend" is not "decide".
    # =================================================================================================================
    def l2_call(self, state: AlertState) -> dict:
        payload: AiRequestPayload = state["request"]
        l1 = state.get("l1") or L1Output()
        questions = ["L2-01", "L2-05", "L2-06"]
        if payload.customer.found and payload.customer.kyc_business:
            questions.append("L2-02")                                  # business fit: only for OUR customer
        if any(e.role == "ON_BEHALF_OF" for e in l1.entities) or any("%" in t for t in state["all_texts"]):
            questions.append("L2-03")                                  # ownership statements
        if state.get("prior_decisions"):
            questions.append("L2-04")                                  # compare with earlier decisions
        customer = None
        if payload.customer.found:
            customer = {"business": payload.customer.kyc_business, "nic_code": payload.customer.nic_code}
        request = L2Request(
            questions=sorted(questions), hit_types=sorted({h.rule_id for h in payload.alert.hits}),
            facts=l1.model_dump(exclude_none=True, exclude_defaults=True), clauses=state["clauses"], texts=state["texts"],
            customer=customer, prior_decisions=state.get("prior_decisions", []), alert_score=state["alert_score"], floors=state["floors"])
        try:
            output, usage = self.d.llm.reason(request)
        except LLMSchemaError:
            log.warning("l2 answer had the wrong shape, trying once more")
            output, usage = self.d.llm.reason(request)
        return {"l2": output, "usage": [*state.get("usage", []), usage.__dict__],
                "audit": _add(state.get("audit", []), "AI_L2_COMPLETED", questions=sorted(questions), tokens=usage.prompt_tokens + usage.completion_tokens)}

    # =================================================================================================================
    # 9. final_gate [CODE]
    #    WHAT: check AI L2's output, then decide whether the AI may propose "AUTO_CLEARED". Otherwise the route is L3.
    #    WHY:  the AI's text is never trusted on its own. AUTO_CLEARED can only be produced HERE, and only if EVERY condition holds.
    # =================================================================================================================
    def final_gate(self, state: AlertState) -> dict:
        settings = self.d.settings
        l2: L2Output = state["l2"]
        payload: AiRequestPayload = state["request"]
        floors = state["floors"]
        l1 = state.get("l1") or L1Output()

        retrieved = {c["clause_id"] for c in state["clauses"]}
        known = self.d.retriever.known_clause_ids()
        clauses_exist = all(c in retrieved and c in known for c in l2.cited_clause_ids)         # the AI may cite only what we showed it
        kyc_text = [payload.customer.kyc_business or ""]
        quotes_ok = all(evidence_found(q, state["all_texts"]) for q in l2.quotes)
        fit = l2.business_fit
        if fit and fit.label != "UNKNOWN":
            quotes_ok = quotes_ok and evidence_found(fit.message_quote, state["all_texts"]) and evidence_found(fit.kyc_quote, kyc_text)
        listed_names = [(h.list_entry or {}).get("name", "") for h in payload.alert.hits if h.list_entry]
        rfi_clean = self._rfi_text_is_clean(l2.draft_text, payload)
        owners = listed_owners(l2.ownership, listed_names)

        # why the AI may NOT clear (empty list = it may)
        blocked: list[str] = []
        if not settings.ai_l2_may_clear:
            blocked.append("ai.l2-may-clear is off")
        if highest_floor_level(floors, self.d.rules.floors) == "L3":
            blocked.append("floor with level L3: " + ",".join(f for f in floors if self.d.rules.floors["floors"].get(f) == "L3"))
        elif floors and not settings.ai_l2_may_clear_floor_l2:
            blocked.append(",".join(floors) + " floor")
        if state["effective_band"] == "L3":
            blocked.append("band L3")
        if any(h.severity == "HARD" for h in payload.alert.hits) or any(h.get("severity") == "HARD" for h in state.get("rescreen_hits", [])):
            blocked.append("HARD hit")
        if state.get("security_event"):
            blocked.append("security event / manipulation")
        for ok, text in ((clauses_exist, "cited clause not in policy"), (quotes_ok, "quote not in message"), (rfi_clean, "RFI text not clean")):
            if not ok:
                blocked.append(text)
        if l2.recommendation != "RECOMMEND_RELEASE" or not l2.cited_clause_ids:
            blocked.append("no RECOMMEND_RELEASE with a cited clause")
        if payload.customer.found and (fit is None or fit.label != "FITS"):
            blocked.append("business fit is not FITS")
        if l2.prior_alert and l2.prior_alert.label == "DIFFERENT_FROM_PRIOR":
            blocked.append("different from prior decision")
        if owners:
            blocked.append("ownership by a listed party")
        if "G-06" in floors:
            blocked.append("G-06 (AI output was invalid)")

        gate = {"schema_valid": True, "clauses_exist": clauses_exist, "quotes_verified": quotes_ok, "rfi_text_clean": rfi_clean,
                "ownership_listed": owners, "clear_allowed": not blocked, "clear_blocked_by": blocked}
        new_floors = floors if (clauses_exist and quotes_ok) else union(floors, [self.d.rules.floors["from_ai"]["output_invalid"]])   # G-06
        route = "AUTO_CLEARED" if not blocked else "L3"
        action = "AI_GATE_PASSED" if (clauses_exist and quotes_ok and rfi_clean) else "AI_GATE_FAILED"
        return {"gate": gate, "proposed_route": route, "floors": new_floors, "audit": _add(state.get("audit", []), action, **{k: v for k, v in gate.items() if k != "ownership_listed"})}

    @staticmethod
    def _rfi_text_is_clean(text: str | None, payload: AiRequestPayload) -> bool:
        """SCP-9.2: a draft RFI / notice must not name a listed party or give the reason for the alert."""
        if not text:
            return True
        low = text.lower()
        names = []
        for h in payload.alert.hits:
            entry = h.list_entry or {}
            names += [entry.get("name", "")] + list(entry.get("aliases") or [])
        if any(n and n.lower() in low for n in names):
            return False
        return not re.search(r"watch ?list|sanction|\bofac\b|\blisted\b|\bscore\b|\bhit\b|alert|S[12]-R\d\d|screening", low)

    # =================================================================================================================
    # 10. emit [CODE]
    #     WHAT: build the ai.review.result payload (status, route, score, bands, floors, attribute table, L1/L2 details, case pack,
    #           versions, errors). The messaging layer publishes it.
    #     WHY:  every path ends here, success or failure, so the orchestrator ALWAYS gets an answer.
    # =================================================================================================================
    def emit(self, state: AlertState) -> dict:
        failed = state.get("failed", False)
        settings, rules = self.d.settings, self.d.rules
        payload: AiRequestPayload | None = state.get("request")
        floors = state.get("floors", payload.alert.floors_so_far if payload else [])
        errors = list(state.get("errors", []))
        l1_ran = bool(state.get("l1_plan"))
        l2: L2Output | None = state.get("l2")

        if failed:
            if "AI_OUTPUT_INVALID" in errors:
                floors = union(floors, [rules.floors["from_ai"]["output_invalid"]])
            route, status, level = "L3", "FAILED", ("L1" if l1_ran else None)
        else:
            status = "COMPLETED"
            level = "L2" if l2 is not None else "L1"
            route = state.get("proposed_route") or "AUTO_CLEARED"      # no L2 ran only when score + floors allow a clear
            if l2 is None and (state["effective_band"] != "AUTO_CLEAR" or floors):
                route = "L3"                                            # safety net: never clear without the checks having passed

        qa = False
        audit = state.get("audit", [])
        if status == "COMPLETED" and route == "AUTO_CLEARED":
            qa = is_qa_sample(state["payment_id"], rules.qa_percent(settings))
            if qa:
                audit = [*audit, {"action": "QA_SAMPLED", "details": {"percent": rules.qa_percent(settings)}}]
        if state.get("security_event"):
            audit = [*audit, {"action": "SECURITY_EVENT", "details": {"floors": floors}}]
        audit = [*audit, {"action": "AI_FAILED" if failed else "AI_RESULT_PUBLISHED", "details": {"route": route, "errors": errors}}]

        result = {
            "status": status,
            "ai_level_reached": level,
            "proposed_route": route,
            "alert_score": None if failed else state.get("alert_score"),
            "raw_band": None if failed else state.get("raw_band"),
            "effective_band": None if failed else state.get("effective_band"),
            "floors_applied": floors,
            "security_event": bool(state.get("security_event")),
            "qa_sample": qa,
            "attribute_table": state.get("attribute_table", []),
            "l1": self._l1_block(state) if l1_ran and not failed else None,
            "l2": self._l2_block(state) if l2 is not None else None,
            "case_pack": self._case_pack(state) if l2 is not None else None,
            "versions": self._versions(state, l2 is not None),
            "errors": errors,
        }
        return {"result": result, "status": status, "proposed_route": route, "qa_sample": qa, "ai_level_reached": level, "audit": audit}

    # ---- helpers that shape the result ------------------------------------------------------------------------------
    @staticmethod
    def _compact_hits(state: AlertState) -> list[dict]:
        return [{"hit_id": h.hit_id, "rule_id": h.rule_id, "original_value": h.original_value, "matched_text": h.matched_text,
                 "field_path": h.field_path} for h in state["request"].alert.hits]

    def _l1_block(self, state: AlertState) -> dict:
        l1 = state.get("l1") or L1Output()
        checks = {}
        for check in state.get("l1_plan", []):
            value = getattr(l1, L1_FIELD_BY_CHECK[check])
            if isinstance(value, list):
                checks[check] = [{**v.model_dump(), "evidence_verified": True} for v in value]
            elif value is not None and hasattr(value, "model_dump"):
                checks[check] = {**value.model_dump(), "evidence_verified": True}
            else:
                checks[check] = value                                  # None or the L1-09 boolean
        rescreen = [{"item": i["value"], "kind": i["kind"], "hits": [h for h in state.get("rescreen_hits", []) if h.get("item_id") == i["item_id"]]}
                    for i in state.get("rescreen_items", [])]
        return {"checks_run": state.get("l1_plan", []), "checks": checks, "code_rechecks": state.get("code_rechecks", {}),
                "rescreen_status": state.get("rescreen_status", "NOT_NEEDED"), "rescreen_results": rescreen,
                "dropped_evidence": state.get("dropped_evidence", [])}

    def _l2_block(self, state: AlertState) -> dict:
        l2: L2Output = state["l2"]
        return {"retrieved_clause_ids": [c["clause_id"] for c in state.get("clauses", [])], "cited_clauses": l2.cited_clause_ids,
                "business_fit": l2.business_fit.model_dump() if l2.business_fit else None,
                "ownership": [o.model_dump() for o in l2.ownership], "prior_alert": l2.prior_alert.model_dump() if l2.prior_alert else None,
                "recommendation": l2.recommendation, "rationale": l2.rationale, "gate": state.get("gate", {})}

    @staticmethod
    def _case_pack(state: AlertState) -> dict:
        l2: L2Output = state["l2"]
        text = {c["clause_id"]: c["text"] for c in state.get("clauses", [])}
        is_rfi = l2.recommendation == "RECOMMEND_RFI"
        return {"summary": l2.rationale, "recommendation": l2.recommendation,
                "cited_clauses": [{"clause_id": c, "text": text.get(c, "")} for c in l2.cited_clause_ids],
                "draft_rfi": l2.draft_text if is_rfi else None, "draft_notice": None if is_rfi else l2.draft_text}

    def _versions(self, state: AlertState, l2_ran: bool) -> dict:
        s = self.d.settings
        payload: AiRequestPayload | None = state.get("request")
        watchlist = next((h.watchlist_version for h in payload.alert.hits if h.watchlist_version), None) if payload else None
        prompts = L1_PROMPT_VERSION + (f" / {L2_PROMPT_VERSION}" if l2_ran else "")
        model = s.openai_model_l1 if s.llm_provider == "openai" else "fake-demo"
        return {"model": model, "prompt": prompts, "watchlist": watchlist,
                "policy": self.d.retriever.version() if l2_ran else None}
