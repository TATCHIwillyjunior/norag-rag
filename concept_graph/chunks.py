"""Chunks the graph is built from.

Shared shape (the same for every retriever): {"chunk_id": str, "text": str, "source": str}.

Two ways to get them:
- load_chunks_jsonl(): chunks exported by the V2/V3 app, so graph chunk IDs are the SAME IDs that
  dense and BM25 return. This is the preferred way, because RRF fuses lists by chunk ID.
- chunk_documents(): stand-alone fallback that repeats the chunking of ask_rag_google.py
  (paragraph filter, fixed window, overlap) on every file of a folder, one file at a time.
"""
import json
from pathlib import Path

CHUNK_SIZE = 800
OVERLAP_PERCENT = 10
MIN_PARAGRAPH_LENGTH = 20


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap_percent=OVERLAP_PERCENT,
               min_paragraph_length=MIN_PARAGRAPH_LENGTH):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_paragraph_length]
    text = "\n\n".join(paragraphs)
    overlap = chunk_size * overlap_percent // 100
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


def chunk_documents(docs_dir, chunk_size=CHUNK_SIZE, overlap_percent=OVERLAP_PERCENT,
                    min_paragraph_length=MIN_PARAGRAPH_LENGTH):
    chunks = []
    for path in sorted(Path(docs_dir).iterdir()):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap_percent, min_paragraph_length)):
            chunks.append({"chunk_id": f"{path.name}::{i}", "text": piece, "source": path.name})
    return chunks


def load_chunks_jsonl(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                chunks.append({"chunk_id": str(row["chunk_id"]), "text": row["text"],
                               "source": row.get("source", "")})
    return chunks


def save_chunks_jsonl(chunks, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
