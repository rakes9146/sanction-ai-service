"""
ask_policy.py - SEE retrieval working, without the graph, the LLM or Kafka.

    uv run python scripts/ask_policy.py "who may block a payment"                  # pure semantic search
    uv run python scripts/ask_policy.py --alert S2-R04 STREET_NAME --format MT103    # the real 3-step ladder for an alert

The second form prints, for each clause, WHY it was chosen: pinned (always), tagged (matches the alert), semantic (filled the rest).
The first run ingests the corpus into data/qdrant if it is empty.
"""
import sys

from ai_service.config import Settings
from ai_service.rag.embedder import make_embedder
from ai_service.rag.ingest import ingest
from ai_service.rag.retriever import PolicyRetriever
from ai_service.rag.store import PolicyStore


def main() -> None:
    settings = Settings()
    settings.check_openai_ready()
    embedder = make_embedder(settings)
    store = PolicyStore(path=None if settings.qdrant_url else settings.qdrant_path, url=settings.qdrant_url or None)
    print(ingest(store, embedder).summary())
    retriever = PolicyRetriever(store, embedder)

    args = sys.argv[1:]
    if args and args[0] == "--alert":
        fmt = None
        if "--format" in args:
            fmt = args[args.index("--format") + 1]
            args = args[: args.index("--format")]
        tags = set(args[1:])
        clauses = retriever.retrieve(tags, fmt, "payment held for review " + " ".join(sorted(tags)))
    else:
        clauses = retriever.ask(" ".join(args) or "who may reject a payment", n=5)

    for c in clauses:
        extra = f"  score={c.score:.3f}" if c.score is not None else ""
        print(f"[{c.why:8}] {c.clause_id:9} {c.title}{extra}\n           {c.text}")


if __name__ == "__main__":
    main()
