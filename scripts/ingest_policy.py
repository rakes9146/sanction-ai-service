"""
ingest_policy.py - load the policy clauses into the vector store.

    uv run python scripts/ingest_policy.py                       # uses src/ai_service/rag/corpus/policy_clauses.json
    uv run python scripts/ingest_policy.py path/to/my_clauses.json

Safe to run again and again: unchanged clauses are skipped, changed ones are updated, removed ones are deleted.
Stop the service first if you use the embedded store (only one process can open the folder).
"""
import sys
from pathlib import Path

from ai_service.config import Settings
from ai_service.rag.embedder import make_embedder
from ai_service.rag.ingest import DEFAULT_CORPUS, ingest
from ai_service.rag.store import PolicyStore


def main() -> None:
    settings = Settings()
    settings.check_openai_ready()
    embedder = make_embedder(settings)
    store = PolicyStore(path=None if settings.qdrant_url else settings.qdrant_path, url=settings.qdrant_url or None)
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS
    report = ingest(store, embedder, corpus)
    print(report.summary())
    print("embedder:", embedder.name, "| store:", settings.qdrant_url or settings.qdrant_path)


if __name__ == "__main__":
    main()
