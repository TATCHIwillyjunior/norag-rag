"""Build the concept graph of a corpus.

  python -m concept_graph.build --docs path/to/kestrel/docs --out graph_out/kestrel
  python -m concept_graph.build --chunks chunks.jsonl     --out graph_out/kestrel   (chunks from the app)

Writes to --out:
  chunks.jsonl       the chunks the graph was built from (chunk_id, text, source)
  graph.json         nodes (type, aliases, chunk_ids) and edges (relation, chunk_id, evidence)
  merges.jsonl       entity-resolution log (normalization, same_as, not_merged_type_conflict)
  rejected.jsonl     every rejected reply/triple with its reason
  build_stats.json   model, LLM calls, tokens, time, counts
  extract_cache.jsonl  raw LLM output per chunk (a rebuild reuses it: no new calls)
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .chunks import chunk_documents, load_chunks_jsonl, save_chunks_jsonl
from .extract import DEFAULT_MODEL, PROMPT_VERSION, Extractor, GeminiLLM
from .graph import build_graph


def run_build(chunks, out_dir, llm, model_name, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    save_chunks_jsonl(chunks, out / "chunks.jsonl")
    ex = Extractor(llm, model_name, cache_path=out / "extract_cache.jsonl")
    per_chunk = []
    for i, c in enumerate(chunks, 1):
        triples = ex.extract(c)
        per_chunk.append((c["chunk_id"], triples))
        log(f"[{i}/{len(chunks)}] {c['chunk_id']}: {len(triples)} triples")
    graph, merges = build_graph(per_chunk)
    elapsed = time.perf_counter() - t0

    (out / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "merges.jsonl", "w", encoding="utf-8") as f:
        for m in merges:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(out / "rejected.jsonl", "w", encoding="utf-8") as f:
        for r in ex.rejected:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    rel_counts = Counter(e["relation"] for e in graph["edges"])
    stats = {
        "model": model_name, "prompt_version": PROMPT_VERSION,
        "chunks": len(chunks),
        "llm_calls": ex.stats["llm_calls"], "cache_hits": ex.stats["cache_hits"],
        "prompt_tokens": ex.stats["prompt_tokens"], "completion_tokens": ex.stats["completion_tokens"],
        "total_tokens": ex.stats["prompt_tokens"] + ex.stats["completion_tokens"],
        "llm_seconds": round(ex.stats["llm_seconds"], 2), "build_seconds": round(elapsed, 2),
        "failed_chunks": ex.stats["failed_chunks"],
        "triples_raw": ex.stats["triples_raw"], "triples_valid": ex.stats["triples_valid"],
        "rejected": dict(Counter(r["reason"] for r in ex.rejected)),
        "entities": len(graph["nodes"]), "relations": len(graph["edges"]),
        "relations_by_type": dict(rel_counts.most_common()),
        "entities_by_type": dict(Counter(n["type"] for n in graph["nodes"]).most_common()),
        "merges": dict(Counter(m["kind"] for m in merges)),
    }
    (out / "build_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return graph, merges, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--docs", help="folder of .md/.txt files (chunked like ask_rag_google.py)")
    src.add_argument("--chunks", help="chunks.jsonl exported by the app (chunk_id, text, source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=10, help="overlap in percent")
    args = ap.parse_args()

    chunks = (load_chunks_jsonl(args.chunks) if args.chunks
              else chunk_documents(args.docs, args.chunk_size, args.overlap))
    _, _, stats = run_build(chunks, args.out, GeminiLLM(args.model), args.model)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
