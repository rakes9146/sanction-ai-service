# Python notes for this project

Every Python / LangGraph idea used in the service, with one example from the code. If a file confuses you, find its idea here.

**Type hints** - `def band_for(score: float, cfg: dict) -> str:` says what goes in and what comes out. Python does not enforce them; your editor and Pydantic use them.

**Dictionaries are everywhere** - JSON becomes a `dict`. Read safely with `.get("key", default)`; `d["key"]` raises `KeyError` if the key is missing.

**Dataclass** - a class that only holds data; Python writes `__init__` for you.
```python
@dataclass
class RescreenOutcome:
    ok: bool
    hits: list[dict]
```

**Pydantic model** - like a dataclass that also *validates*. `Place(role="SOMEWHERE")` raises `ValidationError` because `role` is a `Literal[...]` of allowed values.
That is how we know the AI's answer is valid (check RC-01). `Optional[str]` = a string or `None`. `model_dump()` turns a model back into a dict.
See `models.py`.

**TypedDict** - a dict with known keys. The graph's notebook, `AlertState` (`graph/state.py`). `total=False` = keys may be missing, so always `state.get("key", default)`.

**Settings from the environment** - `config.py`: `class Settings(BaseSettings)`; the attribute `kafka_bootstrap` is filled from the environment variable `KAFKA_BOOTSTRAP`
(or from `.env`). Nothing else in the code reads the environment.

**Protocol / ABC (an "interface")** - `LLM` in `llm/base.py` says "anything with `read` and `reason` is an LLM". `OpenAILLM`, `FakeLLM` and `ScriptedLLM` all are, so the
graph never knows which one it talks to. Same idea for `Embedder`, `Rescreener`, `Repo`.

**f-strings** - `f"score {alert_score:.2f}"` puts values in text; `:.2f` = 2 decimals.

**List comprehension** - `[c["clause_id"] for c in clauses]` = "make a list of the clause ids". Same as a short for-loop.

**Unpacking and `**`** - `{**state, **update}` merges two dicts (later keys win). `[*audit, new]` copies a list and adds one item. The graph nodes use this
instead of modifying lists in place.

**Exceptions** - `raise LLMSchemaError(...)` stops the current function; `try: ... except LLMSchemaError:` catches it. In the graph, `guarded()` (`graph/builder.py`) catches
every error of a step and turns it into `failed=True` plus an error code.

**Context manager (`with`)** - `with attempt:` / `with self._lock:` run a block with setup and cleanup around it (the lock is always released, even after an error).

**Threads, not asyncio** - no `async` / `await` anywhere. The service uses:
* one **thread per Kafka reader** (`messaging/consumers.py`, `threading.Thread`)
* a **thread pool** for alerts (`ThreadPoolExecutor(max_workers=AI_MAX_CONCURRENCY)`): each alert runs the whole graph in one worker thread
* `threading.Event` to let a worker wait for the re-screen answer (`messaging/rescreen.py`): `flag.wait(timeout)` sleeps, `flag.set()` wakes it
* `threading.Lock` to protect the one dictionary that threads share

**Retries** - `tenacity` (`messaging/producer.py`, `llm/openai_llm.py`): "try 3 times, 0.5 s apart" written as `Retrying(stop=stop_after_attempt(3), wait=wait_fixed(0.5))`.

**Logging** - `structlog` writes one JSON object per line. `structlog.contextvars.bind_contextvars(requestId=...)` makes every later line *in this thread* carry the request id.
Never log message text or account numbers.

**Modules and imports** - `from ai_service.checks.scoring import band_for`. The folder `src/ai_service/` is the package; each folder has an `__init__.py`.
`uv sync` installs the project in editable mode so imports work from anywhere.

## LangGraph in five lines
```python
graph = StateGraph(AlertState)                         # 1 a graph with this notebook shape
graph.add_node("prepare", prepare_function)            # 2 a step: takes the notebook, returns the keys it changes
graph.add_edge("a", "b")                               # 3 after a, run b
graph.add_conditional_edges("score", after_score, {"clear": "emit", "needs_l2": "l2_retrieve", "failed": "emit"})  # 4 a fork
app = graph.compile(); result = app.invoke(initial_state)   # 5 run one alert
```
A node returns **only what changed**, e.g. `return {"alert_score": 0.15, "raw_band": "AUTO_CLEAR"}`; LangGraph merges it into the notebook. A fork function returns a
**label**, and the dictionary maps the label to the next node. That is the whole of it. See `docs/graph.md`.

## Commands you will use
| Task | Command |
|---|---|
| install | `uv sync --extra dev` |
| run all tests | `uv run pytest` |
| one test file / one test | `uv run pytest tests/graph/test_scenarios.py` / `uv run pytest -k tc04` |
| see the pipeline on a sample | `uv run python scripts/run_sample.py tehran` |
| load the policy clauses | `uv run python scripts/ingest_policy.py` |
| try retrieval | `uv run python scripts/ask_policy.py --alert S2-R04 STREET_NAME --format MT103` |
| redraw the graph picture | `uv run python scripts/draw_graph.py` |
| start the service | `uv run python -m ai_service.main` |
