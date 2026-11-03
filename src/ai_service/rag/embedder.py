"""
embedder.py - turn text into vectors ("embeddings").

WHAT: an embedder takes a list of texts and returns one list of numbers per text. Texts with similar MEANING get vectors that
      point in a similar direction, which is what lets the retriever find "the clauses about transport documents" for an alert
      about an unclear goods route even if the words are not identical.
WHY two versions:
      OpenAIEmbedder  the real thing (needs OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL).
      HashEmbedder    an OFFLINE stand-in for demos and tests: it only counts shared words. It is not "smart", but with 14 short
                      clauses it ranks well enough to show the whole pipeline working without a key.
RULE: documents and questions MUST be embedded with the SAME embedder. If you switch embedder, run scripts/ingest_policy.py again
      (ingestion notices the size change and rebuilds the collection).
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Protocol

# words that carry no meaning for matching
_STOPWORDS = {"a", "an", "and", "are", "as", "be", "by", "for", "if", "in", "is", "it", "of", "on", "or", "the", "to", "with", "where", "that", "this", "not", "no"}


class Embedder(Protocol):
    """Anything with these three members can be used as an embedder (this is called a 'Protocol' in Python)."""

    name: str      # stored with every vector, so we can tell which embedder made it
    dim: int       # how many numbers per vector

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Offline demo embedder: a bag of words hashed into `dim` buckets, then scaled to length 1."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]
        for word in words:
            bucket = zlib.crc32(word.encode("utf-8")) % self.dim   # the same word always lands in the same bucket
            vector[bucket] += 1.0
        length = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / length for v in vector]


class OpenAIEmbedder:
    """Real embeddings from OpenAI. The model name comes from settings; nothing is hard-coded."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0, base_url: str = "") -> None:
        from openai import OpenAI   # imported here so the offline path works without the key

        self._client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout_s)
        self._model = model
        self.name = f"openai:{model}"
        self.dim = len(self.embed(["dimension probe"])[0])   # ask once to learn the vector size

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def make_embedder(settings) -> Embedder:
    """Pick the embedder from settings.EMBEDDER."""
    if settings.embedder == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model, settings.llm_timeout_s,
                              settings.openai_base_url)
    return HashEmbedder()
