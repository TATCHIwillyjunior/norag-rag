"""GDPR Article 17 purge: every store forgets the document (temp files, no real data touched)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bm25_index
import conversation
import graph as graph_mod
import responsible

GONE, KEPT = "secret-memo.md", "product-guide.md"

GRAPH = {
    "docs": [GONE, KEPT],
    "chunk_hashes": {f"{GONE}#0": "a", f"{KEPT}#0": "b"},
    "nodes": {
        "alice": {"name": "Alice", "chunks": [f"{GONE}#0"]},
        "postroom": {"name": "postroom", "chunks": [f"{GONE}#0", f"{KEPT}#0"]},
        "messaging": {"name": "Messaging", "chunks": [f"{KEPT}#0"]},
    },
    "edges": [
        {"head": "alice", "relation": "member_of", "tail": "messaging", "chunks": [f"{GONE}#0"], "count": 1},
        {"head": "messaging", "relation": "owns", "tail": "postroom", "chunks": [f"{GONE}#0", f"{KEPT}#0"], "count": 2},
    ],
    "stats": {"entities": 3, "relations": 2},
}
CHUNKS_AFTER = [{"id": f"{KEPT}#0", "doc": KEPT, "text": "postroom sends e-mails"}]


class PurgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "graph.json").write_text(json.dumps(GRAPH), encoding="utf-8")
        (self.tmp / "cache.jsonl").write_text(
            json.dumps({"chunk_id": f"{GONE}#0", "triples": []}) + "\n"
            + json.dumps({"chunk_id": f"{KEPT}#0", "triples": []}) + "\n", encoding="utf-8")
        # stale BM25 index that still has the deleted document
        (self.tmp / "bm25.json").write_text(json.dumps({"fingerprint": "old", "entries": {f"{GONE}#0": {}}}))
        self.patches = [
            mock.patch.object(graph_mod, "GRAPH_FILE", self.tmp / "graph.json"),
            mock.patch.object(graph_mod, "CACHE_FILE", self.tmp / "cache.jsonl"),
            mock.patch.object(bm25_index, "INDEX_FILE", self.tmp / "bm25.json"),
            mock.patch.dict(bm25_index._cache, {"fingerprint": None, "index": None}),
            mock.patch.object(responsible, "v2_all_chunks", lambda: CHUNKS_AFTER),
            mock.patch("rag_core.remove_doc"),
            mock.patch("rag_core.CORPUS_DIR", self.tmp),
            mock.patch("rag_core.get_collection"),
        ]
        for p in self.patches:
            p.start()
        import rag_core
        rag_core.get_collection.return_value.get.return_value = {"ids": []}
        session = conversation.get_or_create_session("purgetest")
        session.turns = [
            {"user_question": "q1", "references": [f"{GONE}#0"]},
            {"user_question": "q2", "references": [f"{KEPT}#0"]},
        ]

    def tearDown(self):
        for p in self.patches:
            p.stop()
        conversation.clear_session("purgetest")

    def test_purge_document_clears_every_store(self):
        report = responsible.purge_document(GONE)
        self.assertTrue(report["is_fully_purged"], report)
        g = json.loads((self.tmp / "graph.json").read_text())
        self.assertNotIn("alice", g["nodes"])                       # only in the deleted doc
        self.assertEqual(g["nodes"]["postroom"]["chunks"], [f"{KEPT}#0"])  # shared node kept
        self.assertEqual(len(g["edges"]), 1)
        self.assertEqual(g["edges"][0]["count"], 1)
        self.assertNotIn(GONE, g["docs"])
        self.assertNotIn(f"{GONE}#", (self.tmp / "cache.jsonl").read_text())
        self.assertNotIn(f"{GONE}#", (self.tmp / "bm25.json").read_text())
        turns = conversation.get_or_create_session("purgetest").turns
        self.assertEqual([t["user_question"] for t in turns], ["q2"])
        self.assertEqual(report["removed_from_graph"], {"nodes": 1, "edges": 1, "cache_entries": 1})

    def test_verify_reports_leftovers(self):
        report = responsible.verify_document_purge(GONE)   # nothing purged yet
        self.assertFalse(report["is_fully_purged"])
        self.assertFalse(report["purged_from_graph"])
        self.assertFalse(report["purged_from_bm25"])
        self.assertFalse(report["purged_from_conversations"])


if __name__ == "__main__":
    unittest.main()
