"""Unit tests for the pure functions in ai_service/checks: no model, no database, no Kafka."""
import pytest

from ai_service.checks.attribute_compare import (MATCH, MISMATCH, UNKNOWN, PartyFacts, compare_attributes, compare_identifiers,
                                                 compare_value, facts_for_hit)
from ai_service.checks.evidence import collapse, evidence_found
from ai_service.checks.floors import highest_floor_level, max_level, union
from ai_service.checks.ownership import listed_owners, parse_percent
from ai_service.checks.qa_sampling import is_qa_sample
from ai_service.checks.scoring import (aligned_name_similarity, band_for, corroboration_factor, effective_band, location_score,
                                       name_hit_score)
from ai_service.config import Rules
from ai_service.models import OwnershipStatement

RULES = Rules.load()
SCORING, FLOORS = RULES.scoring, RULES.floors


# ---- PA-02 attribute compare --------------------------------------------------------------------------------------------
def test_compare_value_match_mismatch_unknown():
    assert compare_value("1968-03-21", "1968-03-21") == MATCH
    assert compare_value("Mashhad", "mashhad ") == MATCH           # case and punctuation do not matter
    assert compare_value("1968-03-21", "1991-07-14") == MISMATCH
    assert compare_value(None, "IN") == UNKNOWN                    # one side missing is UNKNOWN, never a mismatch
    assert compare_value("IN", "") == UNKNOWN


def test_compare_identifiers():
    listed = [{"scheme": "PASSPORT", "value": "Z9876543"}]
    assert compare_identifiers(listed, [{"type": "PASSPORT", "value": "z 9876543"}]) == MATCH
    assert compare_identifiers(listed, [{"type": "PASSPORT", "value": "A1111111"}]) == MISMATCH   # same kind, different number
    assert compare_identifiers(listed, [{"type": "LEI", "value": "X"}]) == UNKNOWN                # nothing comparable
    assert compare_identifiers([], [{"type": "PASSPORT", "value": "Z"}]) == UNKNOWN


def test_compare_attributes_row():
    listed = {"dob": "1968-03-21", "nationality": "IR", "city": "Mashhad", "identifiers": []}
    row = compare_attributes(listed, PartyFacts(dob="1991-07-14", nationality="IN", city="Hyderabad"))
    assert row == {"dob": MISMATCH, "nationality": MISMATCH, "city": MISMATCH, "identifiers": UNKNOWN}


PARSED = {"screenable_fields": [{"field_id": "f3", "role": "CREDITOR"}],
          "parties": [{"role": "CREDITOR", "name": "Mohd Abdul Hafiz",
                       "address": {"city": "Pune", "country": "IN"}, "identifiers": [{"type": "DOB", "value": "1990-01-01"}]}]}


def test_facts_come_from_the_message_party_and_the_customer_wins_when_it_is_the_same_person():
    facts = facts_for_hit("f3", PARSED, {"found": False})
    assert (facts.dob, facts.nationality, facts.city) == ("1990-01-01", "IN", "Pune")
    customer = {"found": True, "name": "MOHD ABDUL HAFIZ", "dob": "1991-07-14", "nationality": "IN", "city": "Hyderabad", "identifiers": []}
    assert facts_for_hit("f3", PARSED, customer).city == "Hyderabad"
    other = {**customer, "name": "Somebody Else"}
    assert facts_for_hit("f3", PARSED, other).city == "Pune"       # a different person: the customer record is not used
    assert facts_for_hit("f99", PARSED, {"found": False}) == PartyFacts()


# ---- RC-02 evidence ------------------------------------------------------------------------------------------------------
def test_evidence_must_be_an_exact_piece_of_the_text():
    texts = ["FREIGHT CHARGES CONTAINER 4021 VIA\nTEHRAN STREET HUB DUBAI"]
    assert collapse("a \n  b\t c") == "a b c"
    assert evidence_found("VIA TEHRAN STREET HUB DUBAI", texts)             # a line break counts as a space
    assert not evidence_found("via tehran street", texts)                   # case must match
    assert not evidence_found("VIA TEHRAN AVENUE", texts)                   # invented words
    assert not evidence_found("", texts)
    assert evidence_found("DUBAI Rent", ["...DUBAI", "Rent..."])            # quote across two neighbouring fields


# ---- RC-06/07/09 scoring -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("row, factor", [
    ({"dob": "UNKNOWN", "nationality": "UNKNOWN", "city": "UNKNOWN", "identifiers": "MATCH"}, 1.00),
    ({"dob": "MATCH", "nationality": "MATCH", "city": "UNKNOWN", "identifiers": "UNKNOWN"}, 0.95),
    ({"dob": "MATCH", "nationality": "MISMATCH", "city": "UNKNOWN", "identifiers": "UNKNOWN"}, 0.60),
    ({"dob": "UNKNOWN", "nationality": "UNKNOWN", "city": "UNKNOWN", "identifiers": "UNKNOWN"}, 0.70),
    ({"dob": "MISMATCH", "nationality": "UNKNOWN", "city": "UNKNOWN", "identifiers": "UNKNOWN"}, 0.45),
    ({"dob": "MISMATCH", "nationality": "MISMATCH", "city": "UNKNOWN", "identifiers": "UNKNOWN"}, 0.15),
    ({"dob": "MATCH", "nationality": "MATCH", "city": "MATCH", "identifiers": "MISMATCH"}, 0.10),
])
def test_corroboration_table(row, factor):
    assert corroboration_factor(row, SCORING) == factor


def test_the_two_catalogue_examples():
    tc02 = {"dob": MISMATCH, "nationality": MISMATCH, "city": MISMATCH, "identifiers": UNKNOWN}
    tc06 = {"dob": UNKNOWN, "nationality": MATCH, "city": MATCH, "identifiers": UNKNOWN}
    assert round(name_hit_score(1.00, tc02, SCORING), 2) == 0.15
    assert round(name_hit_score(0.99, tc06, SCORING), 2) == 0.94      # 0.9405


@pytest.mark.parametrize("region, role, score", [
    (True, "PARTY_ADDRESS", 0.90), (False, "UNRELATED", 0.30), (False, "STREET_NAME", 0.30),
    (False, "ROUTING_PORT", 0.55), (None, "STREET_NAME", 0.55), (None, "UNKNOWN", 0.55)])
def test_location_score(region, role, score):
    assert location_score(region, role, SCORING) == score


@pytest.mark.parametrize("score, band", [(0.0, "AUTO_CLEAR"), (0.349, "AUTO_CLEAR"), (0.35, "L2"), (0.849, "L2"), (0.85, "L3"), (1.0, "L3")])
def test_bands(score, band):
    assert band_for(score, SCORING) == band


def test_name_parts_realign_the_comparison():
    same = aligned_name_similarity("MUHAMMAD ABDUL", "HAFEEZ", "Muhammad Abdul Hafeez")
    different = aligned_name_similarity("RAVI", "KUMAR", "Muhammad Abdul Hafeez")
    assert same == pytest.approx(1.0) and different < 0.6
    assert aligned_name_similarity(None, "KUMAR", "Muhammad Hafeez") is None      # no usable parts: no re-score
    assert aligned_name_similarity("RAVI", "KUMAR", "Madonna") is None             # single-word listed name


# ---- floors ---------------------------------------------------------------------------------------------------------------
def test_floors_only_raise_the_level():
    assert highest_floor_level([], FLOORS) == "AUTO_CLEAR"
    assert highest_floor_level(["G-03"], FLOORS) == "L2"
    assert highest_floor_level(["G-03", "G-05"], FLOORS) == "L3"
    assert highest_floor_level(["G-99"], FLOORS) == "L3"                 # an unknown floor is treated as the strictest
    assert max_level("L2", "AUTO_CLEAR", FLOORS) == "L2"
    assert effective_band("AUTO_CLEAR", "L2", FLOORS) == "L2"           # a floor lifts the band ...
    assert effective_band("L3", "L2", FLOORS) == "L3"                    # ... and never lowers it
    assert union(["G-05", "G-01"], ["G-03", "G-01"]) == ["G-01", "G-03", "G-05"]


# ---- QA sampling -----------------------------------------------------------------------------------------------------------
def test_qa_sampling_is_deterministic_and_close_to_the_percentage():
    ids = [f"pay_{i:05d}" for i in range(2000)]
    first = [is_qa_sample(i, 10) for i in ids]
    assert first == [is_qa_sample(i, 10) for i in ids]                   # replaying gives the same answer
    assert 120 <= sum(first) <= 280                                      # about 10 percent of 2000
    assert not any(is_qa_sample(i, 0) for i in ids) and all(is_qa_sample(i, 100) for i in ids)


# ---- ownership --------------------------------------------------------------------------------------------------------------
def test_ownership_the_ai_lists_code_judges():
    assert parse_percent("60%") == 60.0 and parse_percent("holds 12.5 %") == 12.5 and parse_percent("sixty percent") is None
    statements = [OwnershipStatement(owner="Rostek Holdings", owned="X Ltd", percent_text="60%", evidence="e"),
                  OwnershipStatement(owner="Ravi Kumar", owned="Y Ltd", evidence="e")]
    found = listed_owners(statements, ["Rostek Holdings Group"])
    assert [f["owner"] for f in found] == ["Rostek Holdings"] and found[0]["percent"] == 60.0
