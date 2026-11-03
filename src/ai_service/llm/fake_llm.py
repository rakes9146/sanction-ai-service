"""
fake_llm.py - two stand-ins for the model. NEITHER IS AI.

FakeLLM       "demo mode" (LLM_PROVIDER=fake). A handful of simple text rules that imitate what the real model would label, so the
              whole service (Kafka, graph, scoring, gate, RAG) can run end to end on a laptop with no API key.
              It is intentionally simple and will be wrong on text the rules do not cover. NEVER use it to judge real payments.
ScriptedLLM   for unit tests: you hand it the exact answers (or errors) it should give, in order, and it records how often it was called.

Both obey the same LLM interface as the real OpenAILLM, so the graph cannot tell the difference.
"""
from __future__ import annotations

import re
from typing import Optional, Union

from ai_service.llm.base import L1Request, L2Request, LLM
from ai_service.models import (Attribute, Concealment, Entity, Goods, L1Output, L2Output, NameParts, PartyType, Place, TokenRole,
                               Usage)

_ORG_WORDS = {"pvt", "ltd", "llc", "llp", "co", "corp", "fze", "inc", "trading", "logistics", "holdings", "exports", "services",
              "industries", "shipping", "enterprises", "works", "traders", "textiles", "tools"}
_NAME_RULES = {"S1-R01", "S2-R01", "S2-R02", "S2-R03"}
_STREET_WORDS = {"STREET", "ROAD", "ST", "AVENUE", "LANE", "BLVD"}
_PRODUCT_WORDS = {"STEEL", "CARPET", "CARPETS", "RUGS", "KNIVES"}
_SANCTIONED_PLACES = {"YALTA", "SEVASTOPOL", "CRIMEA", "SIMFEROPOL", "PYONGYANG", "DONETSK", "LUHANSK"}
_CONCEALMENT = ("available on request", "do not mention", "details on request", "final buyer", "as per previous")
_INJECTION = ("ignore previous", "ignore all previous", "pre-approved", "mark cleared", "mark as cleared", "set risk", "you are now",
              "system:", "do not flag", "disregard")
_GOODS = {"microcontroller": "8542", "mcu": "8542", "gnss": "8526", "bearing": "8482", "drone": "8806", "servo": "8501", "lng": "2711",
          "crude oil": "2709"}


def _find_text(texts: list[dict], needle: str) -> Optional[str]:
    """The first field value that contains `needle` (case-insensitive). Evidence must be real text, so we return a whole value."""
    for t in texts:
        if needle.lower() in str(t["value"]).lower():
            return str(t["value"])
    return None


class FakeLLM(LLM):
    """Rule-based demo. Not AI."""

    def read(self, request: L1Request) -> tuple[L1Output, Usage]:
        out = L1Output()
        name_hits = [h for h in request.hits if h.get("rule_id") in _NAME_RULES and h.get("original_value")]
        all_text = " ".join(str(t["value"]) for t in request.texts)

        if "L1-02" in request.checks:
            for h in name_hits:
                words = {w.lower().strip(".,") for w in h["original_value"].split()}
                label = "ORGANISATION" if words & _ORG_WORDS else "INDIVIDUAL"
                out.party_types.append(PartyType(subject=h["original_value"], label=label, evidence=h["original_value"]))
        if "L1-03" in request.checks:
            for h in name_hits:
                words = h["original_value"].split()
                if len(words) >= 2 and not ({w.lower().strip(".,") for w in words} & _ORG_WORDS):
                    out.name_parts.append(NameParts(subject=h["original_value"], given=" ".join(words[:-1]).upper(),
                                                    family=words[-1].upper(), evidence=h["original_value"]))
        for h in request.hits:
            token = (h.get("matched_text") or "").strip()
            if h.get("rule_id") != "S2-R04" or not token:
                continue
            value = _find_text(request.texts, token) or ""
            after = re.search(re.escape(token) + r"\s+(\w+)", value, re.IGNORECASE)
            next_word = after.group(1).upper() if after else ""
            before = value.upper().split(token.upper())[0].strip().split()[-1:] if value else []
            label = "STREET_NAME" if next_word in _STREET_WORDS else "PRODUCT" if next_word in _PRODUCT_WORDS else "PLACE"
            if "L1-01" in request.checks:
                out.token_roles.append(TokenRole(token=token.upper(), label=label, evidence=value))
            if "L1-05" in request.checks:
                sanctioned = True if token.upper() in _SANCTIONED_PLACES else False if label != "PLACE" else None
                role = "ROUTING_PORT" if before and before[0] == "VIA" else "UNRELATED" if label != "PLACE" else "UNKNOWN"
                out.places.append(Place(place=f"{token.upper()} {next_word}".strip() if label == "STREET_NAME" else token.upper(), role=role,
                                        country="AE" if "DUBAI" in value.upper() else None, sanctioned_region=sanctioned, evidence=value))
        if "L1-04" in request.checks:
            m = re.search(r"on behalf of ([A-Za-z][A-Za-z0-9&.\- ]{2,50}?)(?= for\b|[,.;]|$)", all_text, re.IGNORECASE)
            if m:
                out.entities.append(Entity(name=m.group(1).strip(), type="ORGANISATION", role="ON_BEHALF_OF", evidence=m.group(0)))
        if "L1-06" in request.checks:
            low = all_text.lower()
            found = [g for g in _GOODS if g in low]
            if found:
                text = _find_text(request.texts, found[0]) or all_text
                out.goods = Goods(goods=found, hs_chapters=sorted({_GOODS[g][:2] for g in found}), dual_use_candidate=True, evidence=text)
        if "L1-07" in request.checks:
            hit = next((p for p in _CONCEALMENT if p in all_text.lower()), None)
            out.concealment = (Concealment(label="CONCEALMENT_INSTRUCTION", evidence=_find_text(request.texts, hit) or "")
                               if hit else Concealment(label="NONE", evidence=""))
        if "L1-08" in request.checks:
            for t in request.texts:
                value = str(t["value"])
                for m in re.finditer(r"IMO\s*:?\s*(\d{7})", value):
                    out.attributes.append(Attribute(type="IMO", value=m.group(1), evidence=m.group(0)))
                for m in re.finditer(r"\bPP\s*:?\s*([A-Z]\d{7})", value):
                    out.attributes.append(Attribute(type="PASSPORT", value=m.group(1), evidence=m.group(0)))
        if "L1-09" in request.checks:
            out.manipulation = any(p in all_text.lower() for p in _INJECTION)
        return out, Usage(step="L1", model="fake-demo")

    def reason(self, request: L2Request) -> tuple[L2Output, Usage]:
        manipulated = bool(request.facts.get("manipulation"))
        if manipulated or request.alert_score >= 0.85:
            recommendation = "RECOMMEND_BLOCK"
        elif request.alert_score < 0.35 and not request.floors:
            recommendation = "RECOMMEND_RELEASE"
        else:
            recommendation = "RECOMMEND_RFI"
        cited = [c["clause_id"] for c in request.clauses if c.get("why") == "tagged"][:2] or [c["clause_id"] for c in request.clauses[:1]]
        quote = str(request.texts[0]["value"]) if request.texts else ""
        rationale = (f"Demo rationale: {len(request.hit_types)} hit type(s) ({', '.join(sorted(set(request.hit_types)))}); "
                     f"score {request.alert_score:.2f}; floors {', '.join(request.floors) or 'none'}.")
        draft = ("Please provide the commercial invoice and the bill of lading for this payment."
                 if recommendation == "RECOMMEND_RFI" else None)
        out = L2Output(cited_clause_ids=cited, recommendation=recommendation, rationale=rationale,
                       quotes=[quote] if quote else [], draft_text=draft)
        return out, Usage(step="L2", model="fake-demo")


class ScriptedLLM(LLM):
    """
    Test double. Give it the answers to return, in order. An item may be an exception instance: it is raised instead.
        llm = ScriptedLLM(l1=[some_L1Output], l2=[some_L2Output])
        llm = ScriptedLLM(l1=[LLMSchemaError("bad"), LLMSchemaError("bad")])     # fails twice
    Calls are counted in `llm.l1_calls` / `llm.l2_calls`, and the last requests are kept for assertions.
    """

    def __init__(self, l1: Optional[list[Union[L1Output, Exception]]] = None, l2: Optional[list[Union[L2Output, Exception]]] = None) -> None:
        self._l1 = list(l1 or [])
        self._l2 = list(l2 or [])
        self.l1_calls = 0
        self.l2_calls = 0
        self.last_l1: Optional[L1Request] = None
        self.last_l2: Optional[L2Request] = None

    def read(self, request: L1Request) -> tuple[L1Output, Usage]:
        self.l1_calls += 1
        self.last_l1 = request
        item = self._l1.pop(0) if self._l1 else L1Output()
        if isinstance(item, Exception):
            raise item
        return item, Usage(step="L1", model="scripted")

    def reason(self, request: L2Request) -> tuple[L2Output, Usage]:
        self.l2_calls += 1
        self.last_l2 = request
        item = self._l2.pop(0) if self._l2 else L2Output(recommendation="RECOMMEND_RFI", rationale="scripted")
        if isinstance(item, Exception):
            raise item
        return item, Usage(step="L2", model="scripted")
