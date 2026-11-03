"""
models.py - every data shape of the service, as Pydantic models.

WHAT: three groups of models
      1. what arrives from the orchestrator (AiRequestPayload)
      2. what the AI model is asked to return (L1Output, L2Output)  <- the "structured output" schemas
      3. small helpers (Usage)
WHY:  Pydantic checks the data for us. If the AI answers with a label that is not on the allowed list, or with the wrong type,
      validation fails and the code knows the answer is invalid (catalogue check RC-01) instead of trusting it.

New to Python? A Pydantic model is a class with typed fields. `Literal["A", "B"]` means "only these exact values".
`Optional[str]` means "a string or None". `model_validate(dict)` builds the object (and raises ValidationError if wrong),
`model_dump()` turns it back into a plain dict.

IMPORTANT: L1Output and L2Output have NO field for a score, a band or a final decision. The AI cannot write one even if it wants to.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# =====================================================================================================================
# 1. REQUEST (orchestrator -> AI service), see references/event-samples/04_ai.review.request.v1.json
# =====================================================================================================================


class Hit(BaseModel):
    """One hit reported by the sanction-check service. Extra fields are kept (extra='allow') so nothing is lost."""

    model_config = ConfigDict(extra="allow")

    hit_id: str
    rule_id: str                              # S1-R01 ... S2-R09
    severity: str = "SOFT"                    # HARD (Service 1) or SOFT (Service 2)
    field_id: Optional[str] = None
    field_path: Optional[str] = None
    original_value: Optional[str] = None      # the text as it was in the payment
    matched_text: Optional[str] = None
    similarity: Optional[float] = None
    watchlist_entry_id: Optional[str] = None
    watchlist_version: Optional[str] = None
    list_entry: Optional[dict] = None         # the listed person / company: name, dob, nationality, city, identifiers
    floor_hint: Optional[str] = None


class Customer(BaseModel):
    """Our own customer (only when `found` is true). Used for attribute compare and the business-fit question."""

    model_config = ConfigDict(extra="allow")

    found: bool = False
    customer_ref: Optional[str] = None
    name: Optional[str] = None
    dob: Optional[str] = None
    nationality: Optional[str] = None
    place_of_birth: Optional[str] = None
    city: Optional[str] = None
    identifiers: list[dict] = Field(default_factory=list)
    kyc_business: Optional[str] = None
    nic_code: Optional[str] = None


class Alert(BaseModel):
    model_config = ConfigDict(extra="allow")

    aggregator_result: str = "ON_HOLD"
    screening_outcome: Optional[str] = None
    hits: list[Hit] = Field(default_factory=list)
    suppressed_hits: list[dict] = Field(default_factory=list)
    floors_so_far: list[str] = Field(default_factory=list)   # floors the orchestrator already knows (G-01 ... G-05)
    pre_ai_checks: dict = Field(default_factory=dict)        # PA-01 / PA-03 (injection regex) / PA-04 (masking)


class AiRequestPayload(BaseModel):
    """The `payload` of an ai.review.request.v1 event."""

    model_config = ConfigDict(extra="allow")

    request_kind: str = "AI_REVIEW"
    timeout_ms: int = 45000
    source_system: Optional[str] = None
    txn_ref: Optional[str] = None
    format: Optional[str] = None
    ai_kill_switch: bool = False
    alert: Alert
    untrusted_message: dict                                   # {"note": ..., "parsed_message": {...masked...}}
    customer: Customer = Field(default_factory=Customer)


# =====================================================================================================================
# 2. AI OUTPUT SCHEMAS. The model must answer in exactly this shape. All labels come from fixed lists.
#    Every answer carries `evidence`: the exact words from the payment the AI relied on. Code checks they really exist (RC-02).
# =====================================================================================================================

# --- AI L1 (read and label), catalogue section 6 ---------------------------------------------------------------------


class TokenRole(BaseModel):
    """L1-01: what is this hit word in its sentence?"""

    token: str
    label: Literal["PLACE", "STREET_NAME", "PRODUCT", "BRAND", "SURNAME", "VESSEL_NAME", "ORGANISATION_NAME", "OTHER", "UNKNOWN"]
    evidence: str


class PartyType(BaseModel):
    """L1-02: is this party a person, a company, a vessel or a bank?"""

    subject: str                                  # the name the answer is about
    label: Literal["INDIVIDUAL", "ORGANISATION", "VESSEL", "FINANCIAL_INSTITUTION", "UNKNOWN"]
    evidence: str


class NameParts(BaseModel):
    """L1-03: the parts of a person's name."""

    subject: str
    given: Optional[str] = None
    family: Optional[str] = None
    patronymic: Optional[str] = None
    honorific: Optional[str] = None
    relation_name: Optional[str] = None
    initials: Optional[str] = None
    evidence: str


class Entity(BaseModel):
    """L1-04: a party / place / reference mentioned in free text, and the role it plays."""

    name: str
    type: Literal["PERSON", "ORGANISATION", "PLACE", "REFERENCE"]
    role: Literal["PAYER", "PAYEE", "ON_BEHALF_OF", "END_USER", "CARRIER", "INVOICE_REF", "UNKNOWN"]
    evidence: str


class Place(BaseModel):
    """L1-05: is a place the party's address, a route, or unrelated? Is it inside a sanctioned region?"""

    place: str
    role: Literal["PARTY_ADDRESS", "ROUTING_PORT", "ORIGIN", "DESTINATION", "UNRELATED", "UNKNOWN"]
    country: Optional[str] = None
    sanctioned_region: Optional[bool] = None      # None = the AI does not know
    evidence: str


class Goods(BaseModel):
    """L1-06: what goods or services are paid for, and could they be dual-use?"""

    goods: list[str] = Field(default_factory=list)
    hs_chapters: list[str] = Field(default_factory=list)
    dual_use_candidate: Optional[bool] = None
    evidence: str


class Concealment(BaseModel):
    """L1-07: wording that hides a party, or fields that contradict each other."""

    label: Literal["CONCEALMENT_INSTRUCTION", "UNDISCLOSED_PRINCIPAL", "PAYER_MISMATCH", "NONE", "UNKNOWN"]
    evidence: str


class Attribute(BaseModel):
    """L1-08: a DOB / passport / ID / IMO written inside free text. CODE compares it, the AI only extracts it."""

    type: Literal["DOB", "PASSPORT", "NATIONAL_ID", "IMO", "LEI", "OTHER"]
    value: str
    evidence: str


class L1Output(BaseModel):
    """
    The complete AI L1 answer. A field stays empty (empty list / None) when that check was not asked.
    Check id -> field:  L1-01 token_roles | L1-02 party_types | L1-03 name_parts | L1-04 entities | L1-05 places
                        L1-06 goods | L1-07 concealment | L1-08 attributes | L1-09 manipulation
    """

    token_roles: list[TokenRole] = Field(default_factory=list)
    party_types: list[PartyType] = Field(default_factory=list)
    name_parts: list[NameParts] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    places: list[Place] = Field(default_factory=list)
    goods: Optional[Goods] = None
    concealment: Optional[Concealment] = None
    attributes: list[Attribute] = Field(default_factory=list)
    manipulation: Optional[bool] = None           # L1-09: does any text try to instruct the screening system?


# Which L1 check writes into which field (used to switch checks on and to build the result)
L1_FIELD_BY_CHECK = {
    "L1-01": "token_roles", "L1-02": "party_types", "L1-03": "name_parts", "L1-04": "entities", "L1-05": "places",
    "L1-06": "goods", "L1-07": "concealment", "L1-08": "attributes", "L1-09": "manipulation",
}

# --- AI L2 (reason with policy), catalogue section 8 ----------------------------------------------------------------


class BusinessFit(BaseModel):
    """L2-02: does the purpose fit our customer's business? Two quotes: one from the payment, one from the KYC text."""

    label: Literal["FITS", "DOES_NOT_FIT", "UNKNOWN"]
    message_quote: str
    kyc_quote: str


class OwnershipStatement(BaseModel):
    """L2-03: an ownership/control statement found in the text. The AI only lists it; CODE does any percentage arithmetic."""

    owner: str
    owned: str
    percent_text: Optional[str] = None            # exactly as written, for example "60%"
    evidence: str


class PriorAlert(BaseModel):
    """L2-04: was a similar alert decided before, and what is different now?"""

    label: Literal["SAME_AS_PRIOR", "DIFFERENT_FROM_PRIOR", "UNKNOWN"]
    difference: str


class L2Output(BaseModel):
    """The complete AI L2 answer. `recommendation` is advice for the human officer. It is NOT a decision."""

    cited_clause_ids: list[str] = Field(default_factory=list)   # L2-01: only ids from the clauses we handed over
    business_fit: Optional[BusinessFit] = None
    ownership: list[OwnershipStatement] = Field(default_factory=list)
    prior_alert: Optional[PriorAlert] = None
    recommendation: Literal["RECOMMEND_RELEASE", "RECOMMEND_RFI", "RECOMMEND_REJECT", "RECOMMEND_BLOCK"]
    rationale: str                                               # at most 120 words
    quotes: list[str] = Field(default_factory=list)              # exact quotes from the payment used in the rationale
    draft_text: Optional[str] = None                             # L2-06: draft RFI / notice, without list names or reasons


# =====================================================================================================================
# 3. SMALL HELPERS
# =====================================================================================================================


@dataclass
class Usage:
    """What one LLM call cost. Stored in the run record so cost and speed can be watched."""

    step: str                    # "L1" or "L2"
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    retries: int = 0


class LLMUnavailable(Exception):
    """The LLM could not be reached (rate limit, timeout, 5xx) even after the retries -> result FAILED / AI_UNAVAILABLE."""


class LLMSchemaError(Exception):
    """The LLM answered, but not in the required shape -> retry once, then FAILED / AI_OUTPUT_INVALID (floor G-06)."""
