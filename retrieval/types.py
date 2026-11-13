"""Types for chunks and results shared by all retrieval methods."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ChunkRecord:
    """A source chunk with a stable ID and provenance."""

    chunk_id: str
    text: str
    source: str
    corpus: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError(f"chunk {self.chunk_id!r} has no text")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError(f"chunk {self.chunk_id!r} has no source")
        if not isinstance(self.corpus, str) or not self.corpus:
            raise ValueError(f"chunk {self.chunk_id!r} has no corpus")
        if not isinstance(self.provenance, Mapping):
            raise ValueError(f"chunk {self.chunk_id!r} provenance must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "corpus": self.corpus,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class RetrievalHit:
    """A ranked chunk returned by a retriever.

    ``path`` is populated by graph retrieval; ``components`` is populated by
    rank fusion and records each input retriever's original rank and score.
    """

    chunk_id: str
    text: str
    source: str
    score: float
    rank: int
    retriever: str
    corpus: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    path: tuple[Mapping[str, Any], ...] = ()
    components: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "score": self.score,
            "rank": self.rank,
            "retriever": self.retriever,
            "corpus": self.corpus,
            "provenance": dict(self.provenance),
        }
        if self.path:
            result["path"] = [dict(edge) for edge in self.path]
        if self.components:
            result["components"] = {
                name: dict(component)
                for name, component in self.components.items()
            }
        return result
