# RAG guide: how the policy clauses reach the AI

**RAG = Retrieval-Augmented Generation.** The model does not know *our* policy. So we (1) keep the policy clauses in a searchable store,
(2) **find** the few that matter for one alert, (3) put them in the prompt. The model must then name the clause ids it used, and code checks the ids.

Code: `src/ai_service/rag/` (five small files, each starts with a comment block that says what and why).

```
INGESTION  (run when the policy changes)                         RETRIEVAL  (for every alert that reaches AI L2)
corpus/policy_clauses.json                                       alert: hit rule ids, labels (STREET_NAME ...), message format
   |  1 LOAD       read + validate the file                         |  step 1  PINNED    always: SCP-2.2, SCP-8.1, SCP-8.2     (filter)
   |  2 CHUNK      one clause = one chunk                           |  step 2  TAGGED    clauses whose tags match the alert   (filter)
   |  3 TEXT       "title. text  tags"                              |  step 3  SEMANTIC  fill the free slots by meaning       (vector search)
   |  4 EMBED      text -> vector (embedder.py)                     |  = at most 5 clauses, each marked why = pinned|tagged|semantic
   |  5 UPSERT     id + vector + payload -> Qdrant                  v
   v  6 CLEAN      delete points no longer in the file             prompt for AI L2  ->  the AI may cite ONLY these ids
Qdrant collection `policy_clauses`                                  final gate: a cited id that was not shown, or does not exist -> blocked + floor G-06
```

## Ingestion (`ingest.py`)
| Step | Function | Why |
|---|---|---|
| Load | `load_corpus()` | reads the JSON, rejects a clause without id / title / text and duplicate ids, so a typo fails loudly |
| Chunk | one clause = one chunk | clauses are short and self-contained and already have ids. **No splitting, no overlap.** If the BA later supplies long documents, split by heading (about 200 words) in this one place |
| Text | `embedding_text()` | title + text + the hit types the clause applies to. Adding the tags lets a question about "S2-R04 place word" also find the clause by meaning |
| Embed | `embedder.embed()` | `OpenAIEmbedder` (real) or `HashEmbedder` (offline demo). Same embedder for documents and questions, always |
| Upsert | `store.upsert()` | point id = `uuid5(clause_id + version)`, so running ingest twice **updates** instead of duplicating |
| Clean | `store.delete()` | whatever is in the store but no longer in the file (or belongs to an old policy version) is removed: the store always equals the file |

Run it: `uv run python scripts/ingest_policy.py`. Output: `policy POC-POLICY-2026-10-01: added 14, updated 0, unchanged 0, removed 0`.
Run it again: `unchanged 14`. Edit one clause: `updated 1`. Change the embedder: the collection is rebuilt (vector size differs).
The service also runs ingest at start-up, so a fresh machine needs no manual step.

Each point carries this metadata (the **payload**): `clause_id, title, text, applies_to_hit_types, applies_to_formats, pinned, version, embedder, text_hash`.
`text_hash` is how ingest knows a clause did not change.

## Retrieval strategy (`retriever.py`): a ladder, strongest rule first

| Step | How it picks | Why this step exists |
|---|---|---|
| 1 PINNED | `pinned == true` filter | a rule that must always be in front of the AI (SCP-8.1: "AI may clear only below 0.35") must never depend on a similarity score |
| 2 TAGGED | `applies_to_hit_types` contains a tag of this alert, and `applies_to_formats` is empty or contains the message format. More matching tags rank first | a banned BIC needs SCP-5.1; a street-name place word needs SCP-6.2. Plain metadata, fully predictable |
| 3 SEMANTIC | vector search with a sentence describing the alert, skipping clauses already chosen | only if there are free slots; it finds helpful clauses nobody tagged (for example SCP-9.2 about the wording of an RFI) |

**Tags of an alert** are the rule ids of the hits (`S2-R04`) plus labels found while processing: `STREET_NAME` (AI read a street), `INJECTION`, `DUAL_USE`,
`CONCEALMENT`. **K = 5** (constant `K` in `retriever.py`): short prompts are cheaper, faster and easier for the model to follow. Pinned clauses are never cut.

**The guard.** `known_clause_ids()` is the set of all ids in the current policy version. The final gate rejects an answer if a cited id was not among the clauses
retrieved for this alert **or** is not in the policy at all. The model cannot invent a clause.

## See it working
```
uv run python scripts/ask_policy.py "who may block a payment"                    # pure semantic search
uv run python scripts/ask_policy.py --alert S2-R04 STREET_NAME --format MT103      # the real 3-step ladder
```
Second command prints three `[pinned]` clauses (SCP-2.2, 8.1, 8.2) then `[tagged]` SCP-6.2 and SCP-6.3, which is exactly the set in `event-samples/08`.
Stop the service first if you use the embedded store (only one process may open `data/qdrant`), or point `QDRANT_URL` at a Qdrant server.

## Embedders and the vector store
* **`HashEmbedder`** (default for demos and tests): counts words into 256 buckets. Not "smart", but with 14 short clauses it ranks well enough to show the pipeline.
* **`OpenAIEmbedder`** (`EMBEDDER=openai`, `OPENAI_EMBEDDING_MODEL=...`): real semantic vectors. Change embedder, then run ingest again.
* **Qdrant embedded** (`QDRANT_PATH=data/qdrant`): a folder on disk, no container. `QDRANT_URL=http://host:6333` uses a server instead. Only `rag/store.py` knows about Qdrant,
  so moving to MongoDB vector search later means rewriting that one file.

## Tuning checklist
| I want to ... | Change |
|---|---|
| show more or fewer clauses | `K` in `retriever.py` |
| always show another clause | set `"pinned": true` in the corpus file, run ingest |
| tie a clause to a kind of hit | add the rule id (or a label such as `STREET_NAME`) to its `applies_to_hit_types` |
| limit a clause to some formats | `applies_to_formats`: `["MT103", "PACS008"]` |
| use the real policy | replace `rag/corpus/policy_clauses.json` (keep the fields), bump `version`, run ingest |
| long policy documents | add a splitter in `ingest.py` between LOAD and TEXT (one chunk per heading) |

## Tests (`tests/unit/test_rag.py`)
ingest twice changes nothing; edited clause = updated, deleted clause = removed; a new policy version replaces the old points; bad corpus is rejected; embedder
size change rebuilds; pinned clauses are present for every alert; tagged order; format filter; at most K and no duplicates; semantic fills free slots;
retrieval never returns an id outside the known set.
