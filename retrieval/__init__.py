"""Shared retrieval contracts and implementations."""

from .bm25 import BM25Index, BM25Retriever, tokenize
from .chunks import chunk_document, chunk_text, load_chunks, load_corpus
from .dense import ChromaDenseRetriever
from .fusion import reciprocal_rank_fusion, retrieve_with_rrf
from .types import ChunkRecord, RetrievalHit

__all__ = [
    "BM25Index",
    "BM25Retriever",
    "ChromaDenseRetriever",
    "ChunkRecord",
    "RetrievalHit",
    "chunk_document",
    "chunk_text",
    "load_chunks",
    "load_corpus",
    "reciprocal_rank_fusion",
    "retrieve_with_rrf",
    "tokenize",
]
