"""
draw_graph.py - draw the LangGraph graph FROM THE CODE, so the picture can never be out of date.

    uv run python scripts/draw_graph.py

Writes:
    docs/graph.mmd            the EXACT graph, as LangGraph sees it (Mermaid text)
    docs/graph.png            its picture (every node and every edge, including all the "failed" arrows)
    docs/graph_overview.png   the picture to LOOK at first: docs/graph_overview.mmd (hand written, colour coded: AI / code / RAG)
                              rendered. A test (tests/unit/test_graph_doc.py) fails if a node is missing from it.
The PNG files need internet: LangGraph asks the mermaid.ink service to render Mermaid text. Without it the .mmd files are still
written and can be pasted into https://mermaid.live.
"""
import base64
from pathlib import Path

import requests

from ai_service.config import Rules, Settings
from ai_service.graph.builder import build_graph
from ai_service.graph.state import Deps, RescreenOutcome
from ai_service.llm.fake_llm import FakeLLM
from ai_service.rag.embedder import HashEmbedder
from ai_service.rag.retriever import PolicyRetriever
from ai_service.rag.store import PolicyStore
from ai_service.repo import MemoryRepo

DOCS = Path(__file__).resolve().parents[1] / "docs"


class _NoRescreen:
    def rescreen(self, *args, **kwargs):
        return RescreenOutcome(ok=True, hits=[], floor_hints=[])


def render_png(mermaid_text: str) -> bytes:
    """Ask the free mermaid.ink service to draw Mermaid text. The text travels in the URL as URL-SAFE base64."""
    encoded = base64.urlsafe_b64encode(mermaid_text.encode("utf-8")).decode("ascii")
    response = requests.get(f"https://mermaid.ink/img/{encoded}", params={"type": "png", "bgColor": "white"}, timeout=60)
    response.raise_for_status()
    return response.content


def main() -> None:
    embedder = HashEmbedder()
    deps = Deps(Settings(_env_file=None), Rules.load(), FakeLLM(), PolicyRetriever(PolicyStore(), embedder), _NoRescreen(), MemoryRepo())
    drawable = build_graph(deps).get_graph()

    (DOCS / "graph.mmd").write_text(drawable.draw_mermaid(), encoding="utf-8")
    print("wrote docs/graph.mmd")
    try:
        (DOCS / "graph.png").write_bytes(render_png((DOCS / "graph.mmd").read_text(encoding="utf-8")))
        print("wrote docs/graph.png")
        (DOCS / "graph_overview.png").write_bytes(render_png((DOCS / "graph_overview.mmd").read_text(encoding="utf-8")))
        print("wrote docs/graph_overview.png")
    except Exception as e:      # noqa: BLE001 - no internet / service down: the text versions are still there
        print("could not render the PNG (", type(e).__name__, "). Paste the .mmd files into https://mermaid.live instead.")


if __name__ == "__main__":
    main()
