"""
ingest.py - INGESTION: get the policy clauses into the vector store.

Run it:   uv run python scripts/ingest_policy.py            (the service also runs it at start-up; it is safe to repeat)

The steps, in order (each one is a short function below):
    1. LOAD      read rag/corpus/policy_clauses.json (or your own file)
    2. CHUNK     one clause = one chunk. Clauses are short and self-contained, so we do NOT split them and do NOT overlap them.
                 (If the BA later gives you long policy documents, this is the function to change: split by heading, ~200 words.)
    3. TEXT      build the text that gets embedded: title + clause text + the hit types it applies to
                 (adding the hit types lets a question like "S2-R04 place word" find the right clause by meaning too)
    4. EMBED     text -> vector (embedder.py)
    5. UPSERT    write id + vector + payload into Qdrant. The id comes from clause_id + version, so re-running UPDATES.
    6. CLEAN     points that are no longer in the file are deleted, so the store always equals the file.

Every point's payload keeps the metadata the retriever filters on:
    clause_id, title, text, applies_to_hit_types, applies_to_formats, pinned, version, embedder, text_hash
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client.models import PointStruct

from ai_service.rag.embedder import Embedder
from ai_service.rag.store import PolicyStore, point_id

DEFAULT_CORPUS = Path(__file__).parent / "corpus" / "policy_clauses.json"


@dataclass
class IngestReport:
    version: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rebuilt_collection: bool = False

    def summary(self) -> str:
        return (f"policy {self.version}: added {len(self.added)}, updated {len(self.updated)}, "
                f"unchanged {len(self.unchanged)}, removed {len(self.removed)}"
                + (" (collection rebuilt: embedder size changed)" if self.rebuilt_collection else ""))


# ---- 1. LOAD ---------------------------------------------------------------------------------------------------------
def load_corpus(path: Path = DEFAULT_CORPUS) -> tuple[str, list[dict]]:
    """Read the JSON file and check every clause has the fields we need. Returns (policy version, clauses)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clauses = data["clauses"]
    seen = set()
    for c in clauses:
        for key in ("clause_id", "title", "text"):
            if not c.get(key):
                raise ValueError(f"clause is missing '{key}': {c}")
        if c["clause_id"] in seen:
            raise ValueError(f"duplicate clause_id {c['clause_id']}")
        seen.add(c["clause_id"])
    return data["version"], clauses


# ---- 2/3. CHUNK + TEXT -----------------------------------------------------------------------------------------------
def embedding_text(clause: dict) -> str:
    """What is turned into a vector for one clause. One clause = one chunk."""
    tags = " ".join(clause.get("applies_to_hit_types", []))
    return f"{clause['title']}. {clause['text']} {tags}".strip()


def _text_hash(clause: dict) -> str:
    """Fingerprint of the clause content: lets us say 'unchanged' without comparing every field."""
    return hashlib.sha256(json.dumps(clause, sort_keys=True).encode("utf-8")).hexdigest()[:16]


# ---- 4/5/6. EMBED, UPSERT, CLEAN -------------------------------------------------------------------------------------
def ingest(store: PolicyStore, embedder: Embedder, path: Path = DEFAULT_CORPUS) -> IngestReport:
    version, clauses = load_corpus(path)
    report = IngestReport(version=version)
    report.rebuilt_collection = store.prepare(embedder.dim)

    existing = store.all_payloads()                      # {point id: payload} already in the store
    wanted_ids = {}
    to_write: list[PointStruct] = []
    vectors = embedder.embed([embedding_text(c) for c in clauses])     # one call for all clauses

    for clause, vector in zip(clauses, vectors):
        pid = point_id(clause["clause_id"], version)
        wanted_ids[pid] = clause["clause_id"]
        payload = {
            "clause_id": clause["clause_id"],
            "title": clause["title"],
            "text": clause["text"],
            "applies_to_hit_types": clause.get("applies_to_hit_types", []),
            "applies_to_formats": clause.get("applies_to_formats", []),
            "pinned": bool(clause.get("pinned", False)),
            "version": version,
            "embedder": embedder.name,
            "text_hash": _text_hash(clause),
        }
        old = existing.get(pid)
        if old is None:
            report.added.append(clause["clause_id"])
        elif old.get("text_hash") != payload["text_hash"] or old.get("embedder") != embedder.name:
            report.updated.append(clause["clause_id"])
        else:
            report.unchanged.append(clause["clause_id"])
            continue                                     # nothing to write
        to_write.append(PointStruct(id=pid, vector=vector, payload=payload))

    store.upsert(to_write)
    stale = [pid for pid in existing if pid not in wanted_ids]       # in the store, but no longer in the file (or an old version)
    report.removed = [existing[pid]["clause_id"] for pid in stale]
    store.delete(stale)
    return report
