"""
store.py - the vector store (Qdrant) for the policy clauses.

WHAT: a thin wrapper around Qdrant with exactly the operations the service needs: prepare the collection, put clauses in,
      take clauses out, find by metadata (scroll), find by meaning (query).
WHY a wrapper: the rest of the code never imports Qdrant. If you later move to another vector database (for example MongoDB
      vector search), only this file changes.

Qdrant terms in one minute:
    collection  a table of points            point   = id + vector + payload
    vector      the numbers made by the embedder     payload = the clause itself (id, title, text, tags, version ...)
    filter      a condition on the payload, for example "pinned == true"
    query       "give me the points whose vectors are closest to this vector", optionally with a filter

Where is the data?  EMBEDDED mode (default): a folder on disk (QDRANT_PATH, default data/qdrant). No server, no container.
Only ONE process can open that folder at a time, so stop the service before running scripts/ingest_policy.py, or set QDRANT_URL
to use a Qdrant server that many processes can share.
"""
from __future__ import annotations

import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointIdsList, PointStruct, VectorParams

COLLECTION = "policy_clauses"


def point_id(clause_id: str, version: str) -> str:
    """The same clause + version always gets the same id, so ingesting twice UPDATES instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"policy:{clause_id}:{version}"))


class PolicyStore:
    def __init__(self, path: Optional[str] = None, url: Optional[str] = None) -> None:
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")      # lives only as long as the process: used by tests

    # ---- setup -------------------------------------------------------------------------------------------------
    def prepare(self, dim: int) -> bool:
        """
        Make sure the collection exists with vectors of size `dim`.
        If it exists with a different size (the embedder changed) it is deleted and rebuilt. Returns True if it was (re)created.
        """
        if self.client.collection_exists(COLLECTION):
            size = self.client.get_collection(COLLECTION).config.params.vectors.size
            if size == dim:
                return False
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        return True

    # ---- write -------------------------------------------------------------------------------------------------
    def upsert(self, points: list[PointStruct]) -> None:
        if points:
            self.client.upsert(COLLECTION, points=points)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.client.delete(COLLECTION, points_selector=PointIdsList(points=ids))

    # ---- read: by metadata ("filter") ---------------------------------------------------------------------------
    def all_payloads(self) -> dict[str, dict]:
        """{point id: payload} for every clause. The corpus is small (tens of clauses) so one page is enough."""
        points, _ = self.client.scroll(COLLECTION, limit=1000, with_payload=True, with_vectors=False)
        return {str(p.id): p.payload for p in points}

    def pinned(self) -> list[dict]:
        """Clauses with pinned == true."""
        flt = Filter(must=[FieldCondition(key="pinned", match=MatchValue(value=True))])
        points, _ = self.client.scroll(COLLECTION, scroll_filter=flt, limit=1000, with_payload=True, with_vectors=False)
        return [p.payload for p in points]

    def tagged(self, tags: list[str]) -> list[dict]:
        """Clauses whose applies_to_hit_types contains at least one of `tags`."""
        if not tags:
            return []
        flt = Filter(must=[FieldCondition(key="applies_to_hit_types", match=MatchAny(any=tags))])
        points, _ = self.client.scroll(COLLECTION, scroll_filter=flt, limit=1000, with_payload=True, with_vectors=False)
        return [p.payload for p in points]

    # ---- read: by meaning ("vector search") ---------------------------------------------------------------------
    def nearest(self, vector: list[float], limit: int, exclude_clause_ids: list[str]) -> list[tuple[dict, float]]:
        """The `limit` clauses closest in meaning to `vector`, skipping the ones already chosen. Returns (payload, score)."""
        flt = Filter(must_not=[FieldCondition(key="clause_id", match=MatchAny(any=exclude_clause_ids))]) if exclude_clause_ids else None
        result = self.client.query_points(COLLECTION, query=vector, query_filter=flt, limit=limit, with_payload=True)
        return [(p.payload, float(p.score)) for p in result.points]
