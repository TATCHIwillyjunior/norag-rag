"""Adapter for the existing Chroma dense-retrieval baseline."""

from collections.abc import Mapping
from typing import Any


class ChromaDenseRetriever:
    """Expose a Chroma collection through the shared retrieval-result shape."""

    name = "dense"

    def __init__(self, collection: Any, *, source: str, corpus: str) -> None:
        if not source or not corpus:
            raise ValueError("source and corpus are required")
        self.collection = collection
        self.source = source
        self.corpus = corpus

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if top_k == 0:
            return []

        response = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = response["ids"][0]
        documents = response["documents"][0]
        if documents is None or len(ids) != len(documents):
            raise ValueError("Chroma returned mismatched IDs and documents")
        distance_rows = response.get("distances") or [[]]
        metadata_rows = response.get("metadatas") or [[]]
        distances = distance_rows[0] or []
        metadatas = metadata_rows[0] or []
        if len(distances) != len(ids):
            raise ValueError("Chroma returned mismatched IDs and distances")
        results: list[dict[str, Any]] = []
        for index, (chunk_id, text) in enumerate(zip(ids, documents), start=1):
            metadata = metadatas[index - 1] if index - 1 < len(metadatas) else None
            if not isinstance(metadata, Mapping):
                metadata = {}
            distance = distances[index - 1] if index - 1 < len(distances) else None
            score = -float(distance) if distance is not None else 0.0
            result_source = metadata.get("source") or self.source
            result_corpus = metadata.get("corpus") or self.corpus
            results.append(
                {
                    "chunk_id": str(chunk_id),
                    "text": str(text),
                    "source": str(result_source),
                    "score": score,
                    "rank": index,
                    "retriever": self.name,
                    "corpus": str(result_corpus),
                    "provenance": {
                        "collection": getattr(self.collection, "name", ""),
                        "distance": distance,
                        "metadata": dict(metadata),
                    },
                }
            )
        return results
