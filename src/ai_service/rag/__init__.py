"""
rag/ - Retrieval-Augmented Generation for the policy clauses (the AI L2 step).

What is RAG, in plain words?
    The AI model does not know OUR bank's policy. Instead of hoping it remembers, we (1) keep the policy clauses in a searchable
    store, (2) FIND the few clauses that matter for this alert, and (3) paste them into the prompt. The AI then answers using
    those clauses and must name the clause ids it used. Code later checks that the ids are real.

The three steps map to three files:
    embedder.py   turns text into a list of numbers (a "vector"); similar meaning => similar numbers
    store.py      a small database for those vectors (Qdrant). Ingestion writes to it, retrieval reads from it
    ingest.py     INGESTION: corpus file -> chunks -> vectors -> store.              Run when the policy changes.
    retriever.py  RETRIEVAL: alert -> the 5 clauses to show the AI.                  Runs for every alert that reaches L2.

Read docs/RAG_GUIDE.md for the picture and the reasons behind each choice.
"""
