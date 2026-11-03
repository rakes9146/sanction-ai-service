"""
builder.py - wires the ten nodes into a LangGraph graph.

LangGraph in five lines (this file uses ONLY these ideas):
    1. `StateGraph(AlertState)`        a graph whose "notebook" has the shape AlertState
    2. `add_node("name", function)`    a step; the function takes the notebook and returns the keys it changes
    3. `add_edge("a", "b")`            after a, always run b
    4. `add_conditional_edges("a", router, {"label": "node", ...})`
                                       after a, call router(state); it returns a label; go to the node mapped to that label
    5. `compile()`                     turn the description into something we can `invoke(state)`

The picture (also generated from this very code by scripts/draw_graph.py into docs/graph.png):

    START → prepare → plan_l1 → l1_call → verify_l1 → rescreen → score ─┬─ clear ───────────────────────────────► emit → END
                                                                         └─ needs_l2 → l2_retrieve → l2_call → final_gate ─► emit
    every step ── failed ──► emit          (a step that fails sets failed=True; `emit` then publishes a FAILED result)
"""
from __future__ import annotations

import time
from typing import Callable

import structlog
from langgraph.graph import END, START, StateGraph

from ai_service.checks.floors import union
from ai_service.graph.nodes import Nodes, _add
from ai_service.graph.state import AlertState, Deps
from ai_service.models import AiRequestPayload, LLMSchemaError, LLMUnavailable
from ai_service.tracing import run_config

log = structlog.get_logger()

# the straight-line part: step -> the step that follows it
NEXT_STEP = {
    "prepare": "plan_l1",
    "plan_l1": "l1_call",
    "l1_call": "verify_l1",
    "verify_l1": "rescreen",
    "rescreen": "score",
    "l2_retrieve": "l2_call",
    "l2_call": "final_gate",
    "final_gate": "emit",
}


# ---- routers: tiny functions that answer "where next?" -----------------------------------------------------------------
def continue_or_fail(state: AlertState) -> str:
    """After most steps: carry on, unless something failed."""
    return "failed" if state.get("failed") else "continue"


def after_score(state: AlertState) -> str:
    """
    After scoring, three possible ways on:
      failed     -> emit (FAILED result)
      clear      -> emit: the score alone says AUTO_CLEAR AND no safety floor applies, so AI L2 is skipped
      needs_l2   -> l2_retrieve: every other alert gets AI L2 and the final gate
    """
    if state.get("failed"):
        return "failed"
    if state.get("effective_band") == "AUTO_CLEAR" and not state.get("floors"):
        return "clear"
    return "needs_l2"


# ---- the safety wrapper: one place that turns exceptions into a FAILED result --------------------------------------------
def guarded(name: str, step: Callable[[AlertState], dict]) -> Callable[[AlertState], dict]:
    """
    Wrap a step so that
      - it is timed (state["timings"][name] in milliseconds), and
      - any error becomes failed=True plus an error code, instead of crashing the whole service.
    Error codes: AI_UNAVAILABLE (model unreachable), AI_OUTPUT_INVALID (wrong shape twice, floor G-06), AI_INTERNAL_ERROR (a bug).
    """

    def run(state: AlertState) -> dict:
        started = time.perf_counter()
        try:
            update = step(state)
        except LLMUnavailable as e:
            update = _failure(state, "AI_UNAVAILABLE", str(e))
        except LLMSchemaError as e:
            update = _failure(state, "AI_OUTPUT_INVALID", str(e))
        except Exception as e:                       # noqa: BLE001 - deliberate catch-all: the orchestrator must always get an answer
            log.exception("graph step crashed", step=name)
            update = _failure(state, "AI_INTERNAL_ERROR", type(e).__name__)   # class name only: never log the message body
        timings = dict(state.get("timings", {}))
        timings[name] = int((time.perf_counter() - started) * 1000)
        return {**update, "timings": timings}

    return run


def _failure(state: AlertState, code: str, detail: str) -> dict:
    log.warning("graph step failed", code=code, detail=detail[:200])
    return {"failed": True, "errors": [*state.get("errors", []), code],
            "audit": _add(state.get("audit", []), "AI_L1_INVALID" if code == "AI_OUTPUT_INVALID" else "AI_FAILED", code=code)}


# ---- build ------------------------------------------------------------------------------------------------------------
def build_graph(deps: Deps):
    nodes = Nodes(deps)
    steps = {
        "prepare": nodes.prepare, "plan_l1": nodes.plan_l1, "l1_call": nodes.l1_call, "verify_l1": nodes.verify_l1,
        "rescreen": nodes.rescreen, "score": nodes.score, "l2_retrieve": nodes.l2_retrieve, "l2_call": nodes.l2_call,
        "final_gate": nodes.final_gate, "emit": nodes.emit,
    }

    graph = StateGraph(AlertState)
    for name, fn in steps.items():
        graph.add_node(name, guarded(name, fn))

    graph.add_edge(START, "prepare")
    for name, following in NEXT_STEP.items():
        graph.add_conditional_edges(name, continue_or_fail, {"continue": following, "failed": "emit"})
    graph.add_conditional_edges("score", after_score, {"clear": "emit", "needs_l2": "l2_retrieve", "failed": "emit"})
    graph.add_edge("emit", END)
    return graph.compile()


def fail_fast(deps: Deps, envelope: dict, code: str) -> dict:
    """For a request so broken that no graph run is possible (for example the payload is not valid): build the FAILED result directly."""
    nodes = Nodes(deps)
    state: AlertState = {"envelope": envelope, "payment_id": envelope.get("payment_id", ""), "request_id": envelope.get("screening_request_id", ""),
                         "failed": True, "errors": [code], "floors": union([], []), "audit": _add([], "AI_FAILED", code=code)}
    return {**state, **nodes.emit(state)}


def run_alert(graph, deps: Deps, envelope: dict) -> AlertState:
    """
    Run ONE alert through the graph and return the final notebook (its `result` key is the ai.review.result payload).
    A request whose payload is not valid never reaches the graph: it gets a FAILED/INVALID_REQUEST result right away.
    """
    try:
        payload = AiRequestPayload.model_validate(envelope["payload"])
    except Exception:                                # noqa: BLE001 - any shape problem means "invalid request"
        return fail_fast(deps, envelope, "INVALID_REQUEST")
    return graph.invoke({"envelope": envelope, "request": payload, "payment_id": envelope.get("payment_id", ""),
                         "request_id": envelope.get("screening_request_id", "")},
                        config=run_config(envelope))           # names the LangSmith trace; does nothing when tracing is off
