"""
Graph tests: the catalogue's test cases run through the REAL graph with a scripted model (record / replay style).
The point of every test is the same: CODE, not the model, decides the outcome.
"""
from ai_service.llm.fake_llm import ScriptedLLM
from ai_service.models import (Attribute, Concealment, Entity, L1Output, L2Output, LLMSchemaError, LLMUnavailable, NameParts, PartyType,
                               Place, TokenRole)
from tests.helpers import StubRescreener, free_text_event, make_deps, run, tc02_event

TEXT_70 = "FREIGHT CHARGES CONTAINER 4021 VIA TEHRAN STREET HUB DUBAI"


def tc02_llm() -> ScriptedLLM:
    """What a good model says for 'Mohd Abdul Hafiz': an individual, given/family parts, nothing hidden."""
    return ScriptedLLM(l1=[L1Output(
        party_types=[PartyType(subject="Mohd Abdul Hafiz", label="INDIVIDUAL", evidence="Mohd Abdul Hafiz")],
        name_parts=[NameParts(subject="Mohd Abdul Hafiz", given="MOHD ABDUL", family="HAFIZ", evidence="Mohd Abdul Hafiz")],
        manipulation=False)])


def test_tc02_different_person_is_auto_cleared_at_l1():
    """NEFT 'Mohd Abdul Hafiz': DOB, nationality and city all differ from the listed person -> factor 0.15 -> score 0.15 -> AUTO_CLEAR."""
    llm = tc02_llm()
    state = run(make_deps(llm), tc02_event())
    r = state["result"]
    assert r["status"] == "COMPLETED"
    assert r["proposed_route"] == "AUTO_CLEARED"
    assert r["ai_level_reached"] == "L1"                  # AI L2 was skipped
    assert r["alert_score"] == 0.15
    assert r["raw_band"] == r["effective_band"] == "AUTO_CLEAR"
    assert r["floors_applied"] == []
    assert r["attribute_table"][0]["dob"] == r["attribute_table"][0]["nationality"] == r["attribute_table"][0]["city"] == "MISMATCH"
    assert r["attribute_table"][0]["identifiers"] == "UNKNOWN"
    assert llm.l1_calls == 1 and llm.l2_calls == 0        # exactly one model call
    assert set(llm.last_l1.checks) == {"L1-02", "L1-03", "L1-08", "L1-09"}      # only the switched-on questions
    assert r["l1"]["checks_run"] == ["L1-02", "L1-03", "L1-08", "L1-09"]
    assert r["l2"] is None and r["case_pack"] is None


def test_tc04_tehran_street_has_floor_g03_so_it_goes_to_l2_then_l3_with_an_rfi_draft():
    """MT103 'VIA TEHRAN STREET HUB DUBAI': the AI says street name, not sanctioned (score 0.30). Floor G-03 raises the band to L2 and a human decides."""
    event = free_text_event(TEXT_70, "S2-R04", "TEHRAN", floors=["G-03"])
    llm = ScriptedLLM(
        l1=[L1Output(token_roles=[TokenRole(token="TEHRAN", label="STREET_NAME", evidence=TEXT_70)],
                     places=[Place(place="TEHRAN STREET, DUBAI", role="ROUTING_PORT", country="AE", sanctioned_region=False, evidence=TEXT_70)],
                     manipulation=False)],
        l2=[L2Output(cited_clause_ids=["SCP-6.2", "SCP-6.3"], recommendation="RECOMMEND_RFI", quotes=[TEXT_70],
                     rationale='Field 70 reads "VIA TEHRAN STREET HUB DUBAI". A Dubai street is not a sanctioned place; ask for transport documents.',
                     draft_text="Please provide the commercial invoice and bill of lading for container 4021.")])
    state = run(make_deps(llm), event)
    r = state["result"]
    assert r["status"] == "COMPLETED"
    assert r["alert_score"] == 0.3
    assert (r["raw_band"], r["effective_band"]) == ("AUTO_CLEAR", "L2")      # the floor lifted it
    assert r["floors_applied"] == ["G-03"]
    assert r["ai_level_reached"] == "L2"
    assert r["proposed_route"] == "L3"
    assert r["l2"]["retrieved_clause_ids"] == ["SCP-2.2", "SCP-8.1", "SCP-8.2", "SCP-6.2", "SCP-6.3"]      # same set as sample 08
    assert r["l2"]["gate"]["clear_allowed"] is False and any("G-03" in b for b in r["l2"]["gate"]["clear_blocked_by"])
    assert r["case_pack"]["recommendation"] == "RECOMMEND_RFI"
    assert r["case_pack"]["draft_rfi"] == "Please provide the commercial invoice and bill of lading for container 4021."
    assert [c["clause_id"] for c in r["case_pack"]["cited_clauses"]] == ["SCP-6.2", "SCP-6.3"]
    assert r["versions"]["policy"] == "POC-POLICY-2026-10-01"
    assert llm.l1_calls == 1 and llm.l2_calls == 1
