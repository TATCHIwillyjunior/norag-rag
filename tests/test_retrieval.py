import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from retrieval.bm25 import BM25Index, BM25Retriever, tokenize
from retrieval.chunks import chunk_text, load_chunks, load_corpus
from retrieval.dense import ChromaDenseRetriever
from retrieval.fusion import reciprocal_rank_fusion, retrieve_with_rrf


class BM25Tests(unittest.TestCase):
    def test_tokenizer_preserves_error_identifiers(self) -> None:
        self.assertIn("e47", tokenize("What does E47 mean?"))

    def test_e47_manual_retrieves_the_exact_code(self) -> None:
        manual = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "vellora-duo-espresso-machine.txt"
        )
        chunks = load_chunks(manual)
        retriever = BM25Retriever(chunks)

        results = retriever.search(
            "What does error E47 mean and how do I fix it?",
            top_k=5,
        )

        self.assertTrue(results)
        self.assertIn("E47", results[0]["text"])
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["corpus"], manual.stem)
        self.assertEqual(results[0]["retriever"], "bm25")

    def test_legacy_search_returns_text_and_score_pairs(self) -> None:
        index = BM25Index(["E47 means grinder blocked", "E48 means low water"])

        results = index.search("E47")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "E47 means grinder blocked")
        self.assertGreater(results[0][1], 0)

    def test_no_overlap_returns_no_irrelevant_results(self) -> None:
        index = BM25Index(["the grinder is blocked", "the water tank is empty"])

        self.assertEqual(index.search("unmatched phrase"), [])

    def test_empty_index_is_safe_and_empty_chunks_are_rejected(self) -> None:
        self.assertEqual(BM25Index([]).search("query"), [])
        with self.assertRaisesRegex(ValueError, "no text"):
            BM25Index([""])

    def test_ties_are_stable_and_ranked_by_chunk_id(self) -> None:
        index = BM25Index(
            [
                {"chunk_id": "z", "text": "needle", "source": "doc", "corpus": "demo"},
                {"chunk_id": "a", "text": "needle", "source": "doc", "corpus": "demo"},
            ]
        )

        first = index.retrieve("needle")
        second = index.retrieve("needle")

        self.assertEqual([item["chunk_id"] for item in first], ["a", "z"])
        self.assertEqual(first, second)

    def test_duplicate_chunk_ids_and_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            BM25Index(
                [
                    {"chunk_id": "same", "text": "one", "source": "doc", "corpus": "demo"},
                    {"chunk_id": "same", "text": "two", "source": "doc", "corpus": "demo"},
                ]
            )
        with self.assertRaises(ValueError):
            BM25Index(["text"], b=1.1)
        with self.assertRaises(ValueError):
            BM25Index(["text"]).search("text", -1)

    def test_idf_and_score_are_finite(self) -> None:
        index = BM25Index(["e47 common", "e48 common", "e49 common"])

        self.assertGreater(index.idf("e47"), index.idf("common"))
        self.assertTrue(math.isfinite(index.score("E47", 0)))


class ChunkingTests(unittest.TestCase):
    def test_fixed_window_chunking_matches_original_settings(self) -> None:
        chunks = chunk_text(
            "abcdefghijklmnop",
            chunk_size=10,
            overlap_percent=20,
            min_paragraph_length=1,
        )

        self.assertEqual(chunks, ["abcdefghij", "ijklmnop"])

    def test_document_chunks_have_stable_ids_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.txt"
            path.write_text("A sufficiently long paragraph for this test.", encoding="utf-8")

            chunks = load_chunks(path, min_paragraph_length=1)

        self.assertEqual(chunks[0].chunk_id, "id_0")
        self.assertEqual(chunks[0].corpus, "corpus")
        self.assertEqual(chunks[0].provenance["chunk_index"], 0)

    def test_multi_file_corpus_ids_match_graph_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "kestrel"
            directory.mkdir()
            (directory / "a.txt").write_text("A sufficiently long paragraph.", encoding="utf-8")
            (directory / "b.md").write_text("Another sufficiently long paragraph.", encoding="utf-8")

            chunks = load_corpus(directory, min_paragraph_length=1, corpus="kestrel")

        self.assertEqual([chunk.chunk_id for chunk in chunks], ["a.txt::0", "b.md::0"])
        self.assertEqual({chunk.corpus for chunk in chunks}, {"kestrel"})


class FusionTests(unittest.TestCase):
    def test_rrf_combines_rankings_and_preserves_component_scores(self) -> None:
        dense = [
            self.result("one", rank=1, score=0.9),
            self.result("two", rank=2, score=0.8),
        ]
        bm25 = [
            self.result("two", rank=1, score=4.2),
            self.result("three", rank=2, score=2.1),
        ]

        fused = reciprocal_rank_fusion(
            {"dense": dense, "bm25": bm25},
            top_k=3,
            rrf_constant=1,
        )

        self.assertEqual([item["chunk_id"] for item in fused], ["two", "one", "three"])
        self.assertEqual(
            fused[0]["components"]["dense"]["rank"],
            2,
        )
        self.assertEqual(fused[0]["components"]["dense"]["score"], 0.8)
        self.assertEqual(fused[0]["components"]["bm25"]["rank"], 1)
        self.assertEqual(fused[0]["components"]["bm25"]["score"], 4.2)
        self.assertAlmostEqual(fused[0]["score"], 5 / 6)
        self.assertEqual([item["rank"] for item in fused], [1, 2, 3])

    def test_duplicate_hits_from_one_retriever_contribute_once(self) -> None:
        duplicate = self.result("one", rank=1, score=1.0)

        fused = reciprocal_rank_fusion(
            {"bm25": [duplicate, duplicate]},
            rrf_constant=0,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["score"], 1.0)

    def test_tied_fusion_scores_have_deterministic_order(self) -> None:
        results = [
            self.result("z", rank=1, score=1.0),
            self.result("a", rank=1, score=1.0),
        ]

        fused = reciprocal_rank_fusion({"bm25": results}, rrf_constant=0)

        self.assertEqual([item["chunk_id"] for item in fused], ["a", "z"])

    def test_same_chunk_id_in_different_corpora_stays_distinct(self) -> None:
        first = self.result("id_0", rank=1, score=1.0, corpus="first")
        second = self.result("id_0", rank=1, score=1.0, corpus="second")

        fused = reciprocal_rank_fusion({"bm25": [first, second]})

        self.assertEqual(len(fused), 2)
        self.assertEqual({item["corpus"] for item in fused}, {"first", "second"})

    def test_graph_results_without_corpus_fuse_with_known_corpus(self) -> None:
        dense = self.result("id_0", rank=1, score=0.8, corpus="kestrel")
        graph = self.result("id_0", rank=1, score=1.0, corpus="")
        graph["source"] = "kestrel.txt"
        graph["path"] = [{"hop": 1, "relation": "HAS"}]

        fused = reciprocal_rank_fusion({"dense": [dense], "graph": [graph]})

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["corpus"], "kestrel")
        self.assertEqual(
            fused[0]["components"]["graph"]["path"],
            [{"hop": 1, "relation": "HAS"}],
        )
        self.assertEqual(
            fused[0]["provenance"]["graph"]["path"],
            [{"hop": 1, "relation": "HAS"}],
        )

    def test_graph_only_fusion_can_supply_its_corpus(self) -> None:
        graph = self.result("id_0", rank=1, score=1.0, corpus="")
        graph["path"] = [{"hop": 1, "relation": "HAS"}]

        fused = reciprocal_rank_fusion(
            {"graph": [graph]},
            default_corpus="kestrel",
        )

        self.assertEqual(fused[0]["corpus"], "kestrel")

    def test_graph_result_without_a_corpus_requires_a_default(self) -> None:
        graph = self.result("id_0", rank=1, score=1.0, corpus="")

        with self.assertRaisesRegex(ValueError, "default_corpus"):
            reciprocal_rank_fusion({"graph": [graph]})

    def test_conflicting_chunk_text_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting"):
            reciprocal_rank_fusion(
                {
                    "dense": [self.result("id_0", rank=1, score=1.0)],
                    "bm25": [
                        self.result("id_0", rank=1, score=1.0, text="different text")
                    ],
                }
            )

    def test_incomplete_result_provenance_is_rejected(self) -> None:
        result = self.result("id_0", rank=1, score=1.0)
        result.pop("provenance")

        with self.assertRaisesRegex(ValueError, "provenance"):
            reciprocal_rank_fusion({"dense": [result]})

    def test_omitted_retrievers_are_not_called(self) -> None:
        class FakeRetriever:
            def __init__(self) -> None:
                self.called = False
                self.last_search: tuple[str, int] | None = None

            def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
                self.called = True
                self.last_search = (query, top_k)
                return [FusionTests.result("one", rank=1, score=1.0)]

        enabled = FakeRetriever()
        disabled = FakeRetriever()
        results = retrieve_with_rrf({"dense": enabled}, "query")

        self.assertEqual(len(results), 1)
        self.assertTrue(enabled.called)
        self.assertEqual(enabled.last_search, ("query", 20))
        self.assertFalse(disabled.called)

    @staticmethod
    def result(
        chunk_id: str,
        *,
        rank: int,
        score: float,
        corpus: str = "demo",
        text: str = "same chunk text",
    ) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "text": text,
            "source": "demo.txt",
            "score": score,
            "rank": rank,
            "retriever": "test",
            "corpus": corpus,
            "provenance": {"chunk_index": 0},
        }


class DenseAdapterTests(unittest.TestCase):
    def test_chroma_adapter_returns_shared_result_shape(self) -> None:
        class FakeCollection:
            name = "crime_800_10_20"

            def query(self, **kwargs: Any) -> dict[str, Any]:
                self.kwargs = kwargs
                return {
                    "ids": [["id_0"]],
                    "documents": [["retrieved text"]],
                    "distances": [[0.25]],
                    "metadatas": [[{"chunk_index": 0}]],
                }

        collection = FakeCollection()
        result = ChromaDenseRetriever(
            collection,
            source="docs/CrimeAndPunishment.txt",
            corpus="CrimeAndPunishment",
        ).search("question", top_k=1)[0]

        self.assertEqual(
            set(result),
            {
                "chunk_id",
                "text",
                "source",
                "score",
                "rank",
                "retriever",
                "corpus",
                "provenance",
            },
        )
        self.assertEqual(result["score"], -0.25)
        self.assertEqual(result["retriever"], "dense")
        self.assertEqual(collection.kwargs["n_results"], 1)


if __name__ == "__main__":
    unittest.main()
