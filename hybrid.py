import inspect
import time

import bm25_index
import fusion
import graph
import rag_core
import v2
from rag_core import RagError, corpus_names, retrieve

RETRIEVERS = ("dense", "bm25", "graph")
CANDIDATES = 20
RRF_K = 60


def hybrid_search(question, docs, top_k, retrievers=("dense",), candidates=CANDIDATES, rrf_k=RRF_K, weights=None):
    retrievers = list(dict.fromkeys(retrievers))
    unknown = [r for r in retrievers if r not in RETRIEVERS]
    if unknown:
        raise RagError(400, "invalid_request", "Unknown retriever: " + ", ".join(unknown))
    if not retrievers:
        raise RagError(400, "invalid_request", "Switch on at least one retriever.")
    docs = docs or corpus_names()
    all_chunks = v2.all_chunks()
    by_id = {c["id"]: c for c in all_chunks}

    lists = {}
    timings = {}
    notes = []
    graph_path = None
    for name in retrievers:
        started = time.time()
        if name == "dense":
            got = retrieve(question, docs, candidates)
        elif name == "bm25":
            got = bm25_index.search(question, docs, candidates, all_chunks)
        else:
            got, graph_path, note = graph.search(question, docs, candidates, by_id)
            if note:
                notes.append(note)
        lists[name] = got
        timings[name] = round((time.time() - started) * 1000)

    ranked = {name: [c["id"] for c in got] for name, got in lists.items()}
    fused = fusion.rrf(ranked, k=rrf_k, weights=weights)
    dense_distance = {c["id"]: c.get("distance") for c in lists.get("dense", [])}
    chunks = []
    for f in fused[:top_k]:
        base = by_id.get(f["id"])
        if base is None:
            continue
        chunks.append({**base, "distance": dense_distance.get(f["id"]), "rrf_score": f["score"], "found_by": f["found_by"]})
    return {
        "chunks": chunks,
        "per_retriever": ranked,
        "graph_path": graph_path,
        "notes": notes,
        "timings_ms": timings,
        "retrievers": retrievers,
    }


def ask_hybrid(question, docs, top_k, retrievers=("dense",), candidates=CANDIDATES, rrf_k=RRF_K):
    if "chunks" not in inspect.signature(rag_core.ask).parameters:
        raise RagError(500, "patch_missing", "rag_core.ask has no 'chunks' parameter. Use the rag_core.py from the V3 pack.")
    started = time.time()
    result = hybrid_search(question, docs, top_k, retrievers, candidates, rrf_k)
    if not result["chunks"]:
        raise RagError(409, "no_index", "No chunk was found. Check the selected documents and retrievers.")
    out = rag_core.ask(question, docs, top_k, chunks=result["chunks"])
    found = {c["id"]: c["found_by"] for c in result["chunks"]}
    out["reference_details"] = [{"id": r, "found_by": found.get(r, {})} for r in out["references"]]
    out["retrieval"] = {
        "retrievers": result["retrievers"],
        "per_retriever": result["per_retriever"],
        "graph_path": result["graph_path"],
        "notes": result["notes"],
        "timings_ms": result["timings_ms"],
    }
    out["total_ms"] = round((time.time() - started) * 1000)
    return out
