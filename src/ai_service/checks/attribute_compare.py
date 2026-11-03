"""
attribute_compare.py - PA-02 "structured attribute compare".

WHAT: for every NAME hit, compare what we know about the person in the payment with the listed person:
      DOB, nationality, city, identifiers.  Each comparison gives MATCH, MISMATCH or UNKNOWN.
WHY:  a name that looks like a listed name is only dangerous if the other facts fit too. "Mohd Abdul Hafiz, born 1991, Hyderabad, India"
      is not "Muhammad Abdul Hafeez, born 1968, Mashhad, Iran". This is CODE, not AI, so it can be trusted and tested.

UNKNOWN means "one side has no value". It is never counted as a mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

MATCH, MISMATCH, UNKNOWN = "MATCH", "MISMATCH", "UNKNOWN"


@dataclass
class PartyFacts:
    """What the payment (or our customer record) says about the party behind a hit."""

    name: str = ""
    dob: Optional[str] = None
    nationality: Optional[str] = None
    city: Optional[str] = None
    identifiers: list[dict] = field(default_factory=list)   # [{"type": "PASSPORT", "value": "Z123"}]


def _clean(value: Optional[str]) -> str:
    return "".join(ch for ch in (value or "").upper() if ch.isalnum())


def compare_value(listed: Optional[str], ours: Optional[str]) -> str:
    """Equal after upper-casing and removing punctuation -> MATCH; both present but different -> MISMATCH; else UNKNOWN."""
    a, b = _clean(listed), _clean(ours)
    if not a or not b:
        return UNKNOWN
    return MATCH if a == b else MISMATCH


def compare_identifiers(listed: list[dict], ours: list[dict]) -> str:
    """
    Listed identifiers look like {"scheme": "PASSPORT", "value": "Z9876543"}; ours like {"type": "PASSPORT", "value": "Z9876543"}.
    Same value -> MATCH.  Same kind of identifier but different value -> MISMATCH.  Otherwise UNKNOWN.
    """
    if not listed or not ours:
        return UNKNOWN
    saw_same_kind = False
    for l in listed:
        for o in ours:
            if _clean(l.get("value")) == _clean(o.get("value")) and _clean(l.get("value")):
                return MATCH
            if _clean(l.get("scheme")) == _clean(o.get("type")):
                saw_same_kind = True
    return MISMATCH if saw_same_kind else UNKNOWN


def compare_attributes(listed_entry: dict, facts: PartyFacts) -> dict:
    """The attribute row for one hit: {"dob": ..., "nationality": ..., "city": ..., "identifiers": ...}."""
    return {
        "dob": compare_value(listed_entry.get("dob"), facts.dob),
        "nationality": compare_value(listed_entry.get("nationality"), facts.nationality),
        "city": compare_value(listed_entry.get("city"), facts.city),
        "identifiers": compare_identifiers(listed_entry.get("identifiers") or [], facts.identifiers),
    }


def facts_for_hit(hit_field_id: Optional[str], parsed_message: dict, customer: dict) -> PartyFacts:
    """
    Find out who the hit is about, using only data that is already in the request.
      1. the hit points at a field (f3); the field has a role (CREDITOR)
      2. the message has a party with that role: its address gives city and country, its identifiers give DOB / ids
      3. if that party is OUR customer (same name), the customer record is more reliable: it wins
    The address COUNTRY is used as the nationality of the party (the message has nothing better).
    """
    role = None
    for f in parsed_message.get("screenable_fields", []):
        if f.get("field_id") == hit_field_id:
            role = f.get("role")
    party = next((p for p in parsed_message.get("parties", []) if p.get("role") == role), None)
    if party is None:
        return PartyFacts()
    address = party.get("address") or {}
    ids = party.get("identifiers") or []
    facts = PartyFacts(
        name=party.get("name") or "",
        dob=next((i.get("value") for i in ids if i.get("type") == "DOB"), None),
        nationality=address.get("country"),
        city=address.get("city"),
        identifiers=[i for i in ids if i.get("type") not in ("DOB", "PLACE_OF_BIRTH", "COUNTRY_OF_BIRTH", "ADDITIONAL")],
    )
    if customer.get("found") and _clean(customer.get("name")) == _clean(facts.name):
        facts = PartyFacts(
            name=facts.name,
            dob=customer.get("dob") or facts.dob,
            nationality=customer.get("nationality") or facts.nationality,
            city=customer.get("city") or facts.city,
            identifiers=[{"type": i.get("scheme") or i.get("type"), "value": i.get("value")}
                         for i in customer.get("identifiers") or []] or facts.identifiers,
        )
    return facts
