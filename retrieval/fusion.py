"""Deterministic Reciprocal Rank Fusion for retrieval result lists."""

from collections.abc import Mapping, Sequence
import math
from typing import Any, Protocol


class SearchableRetriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> Sequence[Mapping[str, Any]]:
        """Return ranked results using the shared retrieval-result shape."""


def _identity(result: Mapping[str, Any]) -> tuple[str, str]:
    chunk_id = result.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("retrieval results must have a non-empty chunk_id")
    corpus = result.get("corpus", "")
    if not isinstance(corpus, str):
        raise ValueError("retrieval result corpus must be a string")
    return corpus, chunk_id


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    top_k: int = 10,
    rrf_constant: int = 60,
    default_corpus: str | None = None,
) -> list[dict[str, Any]]:
    """Fuse only the supplied ranked lists and retain each retriever's evidence.

    The identity key is ``(corpus, chunk_id)`` so IDs such as ``id_0`` can be
    reused safely between separate corpora. A graph result without a corpus
    can inherit the only known corpus for its chunk; graph-only results need
    ``default_corpus``.
    """
    if top_k < 0:
        raise ValueError("top_k must not be negative")
    if rrf_constant < 0:
        raise ValueError("rrf_constant must not be negative")
    if default_corpus == "":
        raise ValueError("default_corpus must not be empty")
    if default_corpus is not None and not isinstance(default_corpus, str):
        raise ValueError("default_corpus must be a string")
    if top_k == 0:
        return []

    known_corpora: dict[str, set[str]] = {}
    for results in ranked_lists.values():
        for result in results:
            if not isinstance(result, Mapping):
                raise TypeError("each retrieval result must be a mapping")
            chunk_id = result.get("chunk_id")
            corpus = result.get("corpus")
            if isinstance(chunk_id, str) and isinstance(corpus, str) and corpus:
                known_corpora.setdefault(chunk_id, set()).add(corpus)
            elif isinstance(chunk_id, str) and default_corpus:
                known_corpora.setdefault(chunk_id, set()).add(default_corpus)

    fused: dict[tuple[str, str], dict[str, Any]] = {}
    seen_by_retriever: dict[str, set[tuple[str, str]]] = {}

    for retriever_name, results in ranked_lists.items():
        if not retriever_name:
            raise ValueError("retriever names must not be empty")
        seen = seen_by_retriever.setdefault(retriever_name, set())
        for position, result in enumerate(results, start=1):
            if not isinstance(result, Mapping):
                raise TypeError("each retrieval result must be a mapping")
            corpus, chunk_id = _identity(result)
            if not corpus:
                candidates = known_corpora.get(chunk_id, set())
                if len(candidates) > 1:
                    raise ValueError(
                        f"result {chunk_id!r} omits corpus but appears in "
                        "multiple corpora"
                    )
                corpus = next(iter(candidates), "")
                if not corpus:
                    raise ValueError(
                        f"result {chunk_id!r} omits corpus; supply default_corpus"
                    )
            key = (corpus, chunk_id)
            if key in seen:
                continue
            seen.add(key)

            text = result.get("text")
            source = result.get("source")
            if not isinstance(text, str) or not text:
                raise ValueError(f"result {chunk_id!r} must include non-empty text")
            if not isinstance(source, str) or not source:
                raise ValueError(f"result {chunk_id!r} must include a source")

            existing = fused.get(key)
            if existing is None:
                existing = {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": source,
                    "corpus": key[0],
                    "score": 0.0,
                    "rank": 0,
                    "retriever": "rrf",
                    "provenance": {},
                    "components": {},
                    "source_variants": [source],
                }
                fused[key] = existing
            elif existing["text"] != text:
                raise ValueError(
                    f"conflicting source data for chunk {chunk_id!r} "
                    f"in corpus {key[0]!r}"
                )
            elif source not in existing["source_variants"]:
                existing["source_variants"].append(source)

            score = result.get("score")
            if (
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not math.isfinite(score)
            ):
                raise ValueError(f"result {chunk_id!r} must include a numeric score")
            original_rank = result.get("rank", position)
            if (
                not isinstance(original_rank, int)
                or isinstance(original_rank, bool)
                or original_rank < 1
            ):
                raise ValueError(f"result {chunk_id!r} must include a positive rank")
            result_provenance = result.get("provenance")
            if result_provenance is None and "path" in result:
                result_provenance = {}
            if not isinstance(result_provenance, Mapping):
                raise ValueError(f"result {chunk_id!r} provenance must be a mapping")
            if not result_provenance and "path" not in result:
                raise ValueError(
                    f"result {chunk_id!r} must include provenance or a graph path"
                )
            contribution: dict[str, Any] = {
                "rank": original_rank,
                "score": score,
                "source": source,
            }
            if "path" in result:
                contribution["path"] = result["path"]
            if result_provenance:
                contribution["provenance"] = result_provenance

            existing["score"] += 1.0 / (rrf_constant + original_rank)
            existing["components"][retriever_name] = contribution
            provenance = dict(result_provenance)
            if "path" in result:
                provenance.setdefault("path", result["path"])
            existing["provenance"][retriever_name] = provenance

    ordered = sorted(
        fused.values(),
        key=lambda result: (
            -result["score"],
            result["corpus"],
            result["chunk_id"],
        ),
    )[:top_k]
    for rank, result in enumerate(ordered, start=1):
        result["score"] = round(result["score"], 12)
        result["rank"] = rank
        if len(result["source_variants"]) > 1:
            result["provenance"]["source_variants"] = result["source_variants"]
        del result["source_variants"]
    return ordered


def retrieve_with_rrf(
    retrievers: Mapping[str, SearchableRetriever],
    query: str,
    *,
    top_k: int = 10,
    candidate_depth: int = 20,
    rrf_constant: int = 60,
    corpus: str | None = None,
) -> list[dict[str, Any]]:
    """Query enabled retrievers and fuse their candidates.

    Disabled retrievers must be omitted from ``retrievers``; omitted
    retrievers are neither called nor represented in the fused result.
    """
    if candidate_depth < 0:
        raise ValueError("candidate_depth must not be negative")
    if top_k < 0:
        raise ValueError("top_k must not be negative")
    if rrf_constant < 0:
        raise ValueError("rrf_constant must not be negative")
    if corpus == "":
        raise ValueError("corpus must not be empty")
    if corpus is not None and not isinstance(corpus, str):
        raise ValueError("corpus must be a string")
    if candidate_depth == 0 or top_k == 0 or not retrievers:
        return []
    ranked_lists = {
        name: retriever.search(query, top_k=candidate_depth)
        for name, retriever in retrievers.items()
    }
    return reciprocal_rank_fusion(
        ranked_lists,
        top_k=top_k,
        rrf_constant=rrf_constant,
        default_corpus=corpus,
    )
