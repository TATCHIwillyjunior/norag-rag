"""Offline tests of the concept graph (fake LLM, no API key, no network).

  python -m pytest tests/test_concept_graph.py -q
"""
import json

from concept_graph.build import run_build
from concept_graph.graph import normalize
from concept_graph.retrieve import GraphRetriever
from concept_graph.triples import sample, summary

# The silver-notebook chain from concepts/concept_graph_challenge_part1.md, one passage per chunk.
CHUNKS = [
    {"chunk_id": "book::1", "source": "book", "text":
        "Denis Amselemovitch had only one treasure: a silver notebook. That winter, he lent it to the "
        "old bookbinder of Stolyarny Lane, who had promised to give it a new leather cover by Easter."},
    {"chunk_id": "book::2", "source": "book", "text":
        "The old bookbinder of Stolyarny Lane was a thin, silent man called Yakov Glinkin."},
    {"chunk_id": "book::3", "source": "book", "text":
        "In the spring, Yakov Glinkin closed his cellar for good and left Petersburg to live with "
        "his brother in Tver."},
    {"chunk_id": "book::4", "source": "book", "text": "He scowled and said nothing for a long time."},
]


def T(s, st, rel, o, ot, ev):
    return {"subject": {"name": s, "type": st}, "relation": rel,
            "object": {"name": o, "type": ot}, "evidence": ev}


REPLIES = {
    "book::1": json.dumps({"triples": [
        T("Denis Amselemovitch", "Person", "HAS", "silver notebook", "Object", "Denis Amselemovitch had only one treasure: a silver notebook"),
        T("silver notebook", "Object", "LENT_TO", "old bookbinder of Stolyarny Lane", "Person", "he lent it to the old bookbinder of Stolyarny Lane"),
        # invalid: unknown relation type -> rejected by the schema
        T("Denis", "Person", "LIKES", "Easter", "Object", "by Easter"),
        # unsupported: the quote is not in the chunk -> rejected by the provenance check
        T("Denis Amselemovitch", "Person", "LOCATED_IN", "Haymarket", "Place", "Denis lived near the Haymarket"),
    ]}),
    "book::2": json.dumps({"triples": [
        T("The old bookbinder of Stolyarny Lane", "Person", "SAME_AS", "Yakov Glinkin", "Person", "The old bookbinder of Stolyarny Lane was a thin, silent man called Yakov Glinkin"),
    ]}),
    "book::3": "```json\n" + json.dumps({"triples": [
        T("Yakov Glinkin", "Person", "MOVED_TO", "Tver", "Place", "left Petersburg to live with his brother in Tver"),
    ]}) + "\n```",
    "book::4": "sorry, I cannot do that",  # malformed twice -> chunk logged as failed
}


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        for cid, reply in REPLIES.items():
            text = next(c["text"] for c in CHUNKS if c["chunk_id"] == cid)
            if text in prompt:
                return reply, {"prompt_tokens": 100, "completion_tokens": 20}
        raise AssertionError("unknown chunk")


def build(tmp_path):
    llm = FakeLLM()
    graph, merges, stats = run_build(CHUNKS, tmp_path, llm, "fake", log=lambda *_: None)
    return llm, graph, merges, stats


def test_validation_and_stats(tmp_path):
    llm, graph, merges, stats = build(tmp_path)
    assert stats["llm_calls"] == 5          # 4 chunks + 1 retry of the malformed one
    assert stats["failed_chunks"] == 1
    assert stats["rejected"] == {"schema": 1, "evidence_not_in_chunk": 1, "malformed_reply": 2}
    assert stats["triples_valid"] == 4
    assert stats["total_tokens"] == 5 * 120
    for e in graph["edges"]:
        assert e["chunk_id"] in {c["chunk_id"] for c in CHUNKS}
    for n in graph["nodes"]:
        assert n["chunk_ids"]


def test_rebuild_uses_cache(tmp_path):
    build(tmp_path)
    llm, _, _, stats = build(tmp_path)
    assert stats["llm_calls"] == 2           # only the failed chunk is asked again (and retried)
    assert stats["cache_hits"] == 3


def test_entity_resolution(tmp_path):
    _, graph, merges, _ = build(tmp_path)
    yakov = [n for n in graph["nodes"] if "Yakov Glinkin" in n["aliases"]]
    assert len(yakov) == 1
    assert "old bookbinder of Stolyarny Lane" in yakov[0]["aliases"]
    assert yakov[0]["merge_chunk_ids"] == ["book::2"]
    assert any(m["kind"] == "same_as" for m in merges)


def test_normalize():
    assert normalize("The Postroom service", "Service") == "postroom"
    assert normalize("Messaging team", "Team") == "messaging"
    assert normalize("Denis's notebook") == "denis notebook"
    assert normalize("KST-5031") == "kst-5031"


def test_two_hop_retrieval(tmp_path):
    build(tmp_path)
    r = GraphRetriever.load(tmp_path / "graph.json", tmp_path / "chunks.jsonl")
    res = r.explain("Where is Denis Amselemovitch's silver notebook now?", top_k=5)
    ids = [h["chunk_id"] for h in res["results"]]
    assert set(ids) == {"book::1", "book::2", "book::3"}   # all three hops, nothing else
    assert ids[0] == "book::1"
    tver = next(h for h in res["results"] if h["chunk_id"] == "book::3")
    assert tver["path"][0]["relation"] == "MOVED_TO" and tver["path"][0]["hop"] == 2
    assert [h["rank"] for h in res["results"]] == [1, 2, 3]
    assert all(h["retriever"] == "graph" for h in res["results"])


def test_no_match_and_one_hop(tmp_path):
    build(tmp_path)
    r = GraphRetriever.load(tmp_path / "graph.json", tmp_path / "chunks.jsonl")
    assert r.explain("What is the refund policy?")["reason"] == "no_entity_match"
    one = GraphRetriever.load(tmp_path / "graph.json", tmp_path / "chunks.jsonl", max_hops=1)
    ids = {h["chunk_id"] for h in one.search("Where is the silver notebook?")}
    assert "book::3" not in ids


def test_triple_sampling(tmp_path):
    build(tmp_path)
    out, rows = sample(tmp_path, n=10, seed=1)
    assert len(rows) == 3                     # fewer edges than 10 in this toy graph
    graded = [dict(r, grade="correct") for r in rows]
    out.write_text("\n".join(json.dumps(r) for r in graded), encoding="utf-8")
    assert summary(tmp_path)["correct"] == 3
