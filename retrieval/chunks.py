"""Shared text chunking and source provenance for dense and lexical indexes."""

from pathlib import Path

from .types import ChunkRecord

CHUNK_SIZE = 800
OVERLAP_PERCENT = 10
MIN_PARAGRAPH_LENGTH = 20


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_percent: int = OVERLAP_PERCENT,
    min_paragraph_length: int = MIN_PARAGRAPH_LENGTH,
) -> list[str]:
    """Apply the project's paragraph filter and fixed-window chunking."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if not 0 <= overlap_percent < 100:
        raise ValueError("overlap_percent must be between 0 and 99")
    if min_paragraph_length < 0:
        raise ValueError("min_paragraph_length must not be negative")

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if len(paragraph.strip()) >= min_paragraph_length
    ]
    normalized_text = "\n\n".join(paragraphs)
    overlap = chunk_size * overlap_percent // 100
    step = chunk_size - overlap
    return [
        normalized_text[start : start + chunk_size]
        for start in range(0, len(normalized_text), step)
    ]


def chunk_document(
    filepath: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap_percent: int = OVERLAP_PERCENT,
    min_paragraph_length: int = MIN_PARAGRAPH_LENGTH,
) -> list[str]:
    """Read a UTF-8 document and return its text chunks."""
    path = Path(filepath)
    return chunk_text(
        path.read_text(encoding="utf-8"),
        chunk_size=chunk_size,
        overlap_percent=overlap_percent,
        min_paragraph_length=min_paragraph_length,
    )


def load_chunks(
    filepath: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap_percent: int = OVERLAP_PERCENT,
    min_paragraph_length: int = MIN_PARAGRAPH_LENGTH,
    corpus: str | None = None,
) -> list[ChunkRecord]:
    """Load one document with IDs and metadata shared by all retrievers.

    IDs remain ``id_0``, ``id_1``, ... to match the existing Chroma index and
    the dense baseline. IDs are unique within a single-document corpus.
    """
    path = Path(filepath)
    corpus_name = path.stem if corpus is None else corpus
    if not corpus_name:
        raise ValueError("corpus must not be empty")
    settings = {
        "chunk_size": chunk_size,
        "overlap_percent": overlap_percent,
        "min_paragraph_length": min_paragraph_length,
    }
    return [
        ChunkRecord(
            chunk_id=f"id_{index}",
            text=text,
            source=str(path),
            corpus=corpus_name,
            provenance={"chunk_index": index, "chunking": settings},
        )
        for index, text in enumerate(
            chunk_document(
                path,
                chunk_size=chunk_size,
                overlap_percent=overlap_percent,
                min_paragraph_length=min_paragraph_length,
            )
        )
    ]


def load_corpus(
    docs_directory: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap_percent: int = OVERLAP_PERCENT,
    min_paragraph_length: int = MIN_PARAGRAPH_LENGTH,
    corpus: str | None = None,
) -> list[ChunkRecord]:
    """Load a directory of text/Markdown files with graph-compatible IDs."""
    directory = Path(docs_directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"corpus directory not found: {directory}")
    corpus_name = directory.name if corpus is None else corpus
    if not corpus_name:
        raise ValueError("corpus must not be empty")

    records: list[ChunkRecord] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        for index, text in enumerate(
            chunk_document(
                path,
                chunk_size=chunk_size,
                overlap_percent=overlap_percent,
                min_paragraph_length=min_paragraph_length,
            )
        ):
            records.append(
                ChunkRecord(
                    chunk_id=f"{path.name}::{index}",
                    text=text,
                    source=path.name,
                    corpus=corpus_name,
                    provenance={
                        "chunk_index": index,
                        "chunking": {
                            "chunk_size": chunk_size,
                            "overlap_percent": overlap_percent,
                            "min_paragraph_length": min_paragraph_length,
                        },
                    },
                )
            )
    return records
