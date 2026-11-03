"""RAG tests: ingestion (idempotent, updates, removals, rebuild) and the retrieval ladder (pinned -> tagged -> semantic)."""
import json

import pytest

from ai_service.rag.embedder import HashEmbedder
from ai_service.rag.ingest import embedding_text, ingest, load_corpus
from ai_service.rag.retriever import PolicyRetriever
from ai_service.rag.store import PolicyStore


@pytest.fixture()
def rag():
    embedder = HashEmbedder()
    store = PolicyStore()                                   # in memory
    report = ingest(store, embedder)
    return store, embedder, PolicyRetriever(store, embedder), report


# ---- embedder -----------------------------------------------------------------------------------------------------------------
def test_hash_embedder_is_deterministic_normalised_and_meaningful():
    e = HashEmbedder()
    a, b, c = e.embed(["request transport documents", "request transport documents", "banned intermediary bank"])
    assert a == b and len(a) == e.dim == 256
    assert sum(x * x for x in a) == pytest.approx(1.0)
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert dot(a, b) > dot(a, c)                            # similar text points the same way


# ---- ingestion ------------------------------------------------------------------------------------------------------------------
def test_ingest_loads_all_14_clauses_once(rag):
    store, embedder, retriever, report = rag
    assert len(report.added) == 14 and not report.updated and not report.removed and report.rebuilt_collection
    assert len(store.all_payloads()) == 14
    assert retriever.version() == "POC-POLICY-2026-10-01"
    assert {"SCP-2.2", "SCP-8.1", "SCP-8.2", "SCP-9.2", "SCP-6.2"} <= retriever.known_clause_ids()


def test_ingest_twice_changes_nothing(rag):
    store, embedder, _, _ = rag
    again = ingest(store, embedder)
    assert len(again.unchanged) == 14 and not again.added and not again.updated and not again.removed
    assert len(store.all_payloads()) == 14                  # no duplicates


def test_ingest_updates_changed_clauses_and_removes_deleted_ones(tmp_path, rag):
    store, embedder, retriever, _ = rag
    version, clauses = load_corpus()
    clauses = [c for c in clauses if c["clause_id"] != "SCP-7.1"]                        # remove one
    clauses[0] = {**clauses[0], "text": clauses[0]["text"] + " (edited)"}                # change one
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"version": version, "clauses": clauses}), encoding="utf-8")
    report = ingest(store, embedder, path)
    assert report.removed == ["SCP-7.1"] and report.updated == [clauses[0]["clause_id"]] and len(report.unchanged) == 12
    assert "SCP-7.1" not in retriever.known_clause_ids()


def test_a_new_policy_version_replaces_the_old_points(tmp_path, rag):
    store, embedder, retriever, _ = rag
    version, clauses = load_corpus()
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"version": "POC-POLICY-2027-01-01", "clauses": clauses}), encoding="utf-8")
    report = ingest(store, embedder, path)
    assert len(report.added) == 14 and len(report.removed) == 14       # new ids (version is part of the id), old ones deleted
    assert retriever.version() == "POC-POLICY-2027-01-01" and len(store.all_payloads()) == 14


def test_corpus_validation_rejects_a_clause_without_text(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": "v", "clauses": [{"clause_id": "SCP-1", "title": "t", "text": ""}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        load_corpus(path)
    dup = {"clause_id": "SCP-1", "title": "t", "text": "x"}
    path.write_text(json.dumps({"version": "v", "clauses": [dup, dup]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(path)


def test_embedding_text_contains_title_text_and_tags():
    text = embedding_text({"title": "T", "text": "body", "applies_to_hit_types": ["S2-R04"]})
    assert "T" in text and "body" in text and "S2-R04" in text


def test_changing_the_embedder_size_rebuilds_the_collection():
    store = PolicyStore()
    ingest(store, HashEmbedder(dim=64))
    report = ingest(store, HashEmbedder(dim=128))
    assert report.rebuilt_collection and len(report.added) == 14


# ---- retrieval -------------------------------------------------------------------------------------------------------------------
def ids(clauses):
    return [c.clause_id for c in clauses]


def test_pinned_clauses_are_always_present_whatever_the_alert(rag):
    _, _, retriever, _ = rag
    for tags in (set(), {"S2-R04"}, {"S1-R02"}, {"UNKNOWN-TAG"}):
        chosen = retriever.retrieve(tags, "MT103", "whatever")
        assert {"SCP-2.2", "SCP-8.1", "SCP-8.2"} <= set(ids(chosen))
        assert all(c.why == "pinned" for c in chosen if c.clause_id in {"SCP-2.2", "SCP-8.1", "SCP-8.2"})


def test_tagged_clauses_follow_the_hit_type_and_the_best_match_comes_first(rag):
    _, _, retriever, _ = rag
    chosen = retriever.retrieve({"S2-R04", "STREET_NAME"}, "MT103", "place word")
    assert ids(chosen) == ["SCP-2.2", "SCP-8.1", "SCP-8.2", "SCP-6.2", "SCP-6.3"]      # the set shown in sample 08
    assert [c.why for c in chosen] == ["pinned", "pinned", "pinned", "tagged", "tagged"]


def test_format_filter_drops_clauses_for_other_formats(rag):
    _, _, retriever, _ = rag
    neft = retriever.retrieve({"S2-R04"}, "NEFT", "")                   # SCP-6.3 lists only MT103 / PACS008
    assert "SCP-6.3" not in ids(neft) and "SCP-6.2" in ids(neft)


def test_at_most_k_clauses_and_never_a_duplicate(rag):
    _, _, retriever, _ = rag
    chosen = retriever.retrieve({"S2-R01", "S2-R04", "S1-R02", "INJECTION", "S2-R08"}, "PACS008", "banned bank and place and goods")
    assert len(chosen) == 5 and len(set(ids(chosen))) == 5


def test_semantic_search_fills_the_free_slots(rag):
    _, _, retriever, _ = rag
    chosen = retriever.retrieve({"S1-R01"}, "RTGS", "the wording of a request for information must not mention list names")
    semantic = [c for c in chosen if c.why == "semantic"]
    assert len(chosen) == 5 and semantic and "SCP-9.2" in [c.clause_id for c in semantic]
    assert all(c.score is not None for c in semantic)


def test_ask_is_pure_semantic_search(rag):
    _, _, retriever, _ = rag
    top = retriever.ask("who may block or reject a payment", n=2)
    assert top[0].clause_id == "SCP-2.2" and all(c.why == "semantic" for c in top)


def test_retrieval_never_returns_an_id_outside_the_known_set(rag):
    _, _, retriever, _ = rag
    known = retriever.known_clause_ids()
    for tags in ({"S2-R04"}, {"S1-R02"}, {"INJECTION"}, set()):
        assert set(ids(retriever.retrieve(tags, "MT103", "anything"))) <= known
