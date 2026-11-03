"""
scoring.py - RC-06, RC-07, RC-09: turn verified facts into a score and a band.

WHAT: the numbers. A hit gets a score between 0 and 1, the alert gets the HIGHEST hit score, and the score picks a band:
        score < 0.35       AUTO_CLEAR
        0.35 <= score < 0.85   L2
        score >= 0.85      L3
WHY:  the AI has no say here. The AI only supplied labels; this file (with numbers from config/scoring.yml) does the arithmetic,
      so the same facts always give the same score.

Worked examples (from the catalogue):
    TC-02  similarity 1.00 x factor 0.15 (3 mismatches)            = 0.15  -> AUTO_CLEAR
    TC-06  similarity 0.99 x factor 0.95 (country+city match)      = 0.94  -> L3
"""
from __future__ import annotations

from typing import Optional

from rapidfuzz.distance import JaroWinkler

from ai_service.checks.attribute_compare import MATCH, MISMATCH
from ai_service.checks.floors import level_rank


def corroboration_factor(row: dict, cfg: dict) -> float:
    """
    How much the other attributes support or weaken the name match (table in config/scoring.yml).
    `row` is the attribute row of one hit: {"dob": "MISMATCH", "nationality": ..., "city": ..., "identifiers": ...}
    """
    c = cfg["corroboration"]
    if row.get("identifiers") == MATCH:
        return c["identifier_match"]
    if row.get("identifiers") == MISMATCH:
        return c["identifier_mismatch"]
    values = [row.get("dob"), row.get("nationality"), row.get("city"), row.get("identifiers")]
    matches, mismatches = values.count(MATCH), values.count(MISMATCH)
    if matches and not mismatches:
        return c["match_no_mismatch"]
    if matches and mismatches:
        return c["match_and_mismatch"]
    if not matches and not mismatches:
        return c["all_unknown"]
    return c["one_mismatch"] if mismatches == 1 else c["two_or_more_mismatch"]


def name_hit_score(similarity: float, row: dict, cfg: dict) -> float:
    """Score of a name hit = similarity x corroboration factor."""
    return similarity * corroboration_factor(row, cfg)


def location_score(sanctioned_region: Optional[bool], role: str, cfg: dict) -> float:
    """
    RC-07: score of a place hit, from what the AI read (L1-05).
      sanctioned_region true                                -> 0.90
      role UNRELATED or STREET_NAME, region false           -> 0.30
      anything else (unknown, conflicting)                  -> 0.55
    """
    loc = cfg["location"]
    if sanctioned_region is True:
        return loc["sanctioned_region_true"]
    if sanctioned_region is False and role in ("UNRELATED", "STREET_NAME"):
        return loc["unrelated_or_street_not_sanctioned"]
    return loc["unknown_or_conflicting"]


def aligned_name_similarity(given: Optional[str], family: Optional[str], listed_name: str) -> Optional[float]:
    """
    RC-06: re-score a name using the PARTS the AI found, so given name is compared with given name and family with family.
    The listed name is split the simple way: last word = family, the rest = given.  Returns None if the AI gave no usable parts.
    """
    if not given or not family or not listed_name.strip():
        return None
    words = listed_name.upper().split()
    if len(words) < 2:
        return None
    listed_given, listed_family = " ".join(words[:-1]), words[-1]
    given_sim = JaroWinkler.similarity(given.upper(), listed_given)
    family_sim = JaroWinkler.similarity(family.upper(), listed_family)
    return (given_sim + family_sim) / 2


def band_for(score: float, cfg: dict) -> str:
    """RC-09: the band the score alone gives."""
    b = cfg["bands"]
    if score < b["auto_clear_below"]:
        return "AUTO_CLEAR"
    if score < b["l3_from"]:
        return "L2"
    return "L3"


def effective_band(raw_band: str, floor_level_name: str, floors_cfg: dict) -> str:
    """The strictest of the score band and the level every floor demands."""
    return raw_band if level_rank(raw_band, floors_cfg) >= level_rank(floor_level_name, floors_cfg) else floor_level_name
