"""In-memory BM25 retrieval with a reusable, dependency-free index."""

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .types import ChunkRecord, RetrievalHit

BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word-like tokens, preserving codes like E47."""
    return _TOKEN_PATTERN.findall(text.casefold())


def _coerce_chunk(value: str | ChunkRecord | Mapping[str, Any], index: int) -> ChunkRecord:
    if isinstance(value, ChunkRecord):
        return value
    if isinstance(value, str):
        return ChunkRecord(
            chunk_id=f"id_{index}",
            text=value,
            source="",
            corpus="",
        )
    if isinstance(value, Mapping):
        return ChunkRecord(
            chunk_id=str(value.get("chunk_id", f"id_{index}")),
            text=str(value.get("text", "")),
            source=str(value.get("source", "unknown")),
            corpus=str(value.get("corpus", "unknown")),
            provenance=value.get("provenance", {}),
        )
    raise TypeError(f"unsupported chunk type: {type(value).__name__}")


class BM25Index:
    """BM25 index; ``search`` retains the original (text, score) API."""

    def __init__(
        self,
        chunks: Sequence[str | ChunkRecord | Mapping[str, Any]],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        source: str = "unknown",
        corpus: str = "unknown",
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.k1 = k1
        self.b = b
        self.chunks = [
            self._chunk_with_defaults(chunk, index, source, corpus)
            for index, chunk in enumerate(chunks)
        ]
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id values must be unique within a BM25 index")

        self.word_counts = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.chunk_lengths = [sum(counts.values()) for counts in self.word_counts]
        self.average_length = (
            sum(self.chunk_lengths) / len(self.chunk_lengths)
            if self.chunk_lengths
            else 0.0
        )
        self.chunk_frequency: Counter[str] = Counter()
        for counts in self.word_counts:
            self.chunk_frequency.update(counts.keys())

    @staticmethod
    def _chunk_with_defaults(
        value: str | ChunkRecord | Mapping[str, Any],
        index: int,
        source: str,
        corpus: str,
    ) -> ChunkRecord:
        if isinstance(value, str):
            return ChunkRecord(
                chunk_id=f"id_{index}",
                text=value,
                source=source,
                corpus=corpus,
                provenance={"chunk_index": index},
            )
        chunk = _coerce_chunk(value, index)
        if source != "unknown" and (
            chunk.source == "unknown" or chunk.corpus == "unknown"
        ):
            return ChunkRecord(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=source if chunk.source == "unknown" else chunk.source,
                corpus=corpus if chunk.corpus == "unknown" else chunk.corpus,
                provenance=chunk.provenance,
            )
        return chunk

    def idf(self, word: str) -> float:
        """Return positive BM25 IDF, including for unseen query terms."""
        count = len(self.chunks)
        frequency = self.chunk_frequency[word]
        return math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))

    def score(self, query: str, index: int) -> float:
        if not 0 <= index < len(self.chunks):
            raise IndexError("chunk index is out of range")
        counts = self.word_counts[index]
        length = self.chunk_lengths[index]
        length_ratio = length / self.average_length if self.average_length else 0.0
        total = 0.0
        for word in dict.fromkeys(tokenize(query)):
            term_frequency = counts[word]
            if not term_frequency:
                continue
            denominator = term_frequency + self.k1 * (
                1.0 - self.b + self.b * length_ratio
            )
            total += self.idf(word) * (
                term_frequency * (self.k1 + 1.0) / denominator
            )
        return total

    def ranked_hits(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if top_k == 0 or not self.chunks:
            return []

        scored = [
            (self.score(query, index), chunk)
            for index, chunk in enumerate(self.chunks)
        ]
        scored = [(score, chunk) for score, chunk in scored if score > 0.0]
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalHit(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                score=score,
                rank=rank,
                retriever="bm25",
                corpus=chunk.corpus,
                provenance=chunk.provenance,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [hit.to_dict() for hit in self.ranked_hits(query, top_k)]

    def search(self, query: str, n_results: int = 5) -> list[tuple[str, float]]:
        """Backward-compatible results for the original BM25 example."""
        return [
            (hit.text, hit.score)
            for hit in self.ranked_hits(query, n_results)
        ]


class BM25Retriever:
    """Structured-result adapter implementing the shared ``search`` contract."""

    name = "bm25"

    def __init__(
        self,
        chunks: Sequence[str | ChunkRecord | Mapping[str, Any]] | BM25Index,
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        source: str = "unknown",
        corpus: str = "unknown",
    ) -> None:
        self.index = (
            chunks
            if isinstance(chunks, BM25Index)
            else BM25Index(chunks, k1=k1, b=b, source=source, corpus=corpus)
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.index.retrieve(query, top_k)
