import inspect
import time
from typing import Any

import bm25_index
import conversation
import fusion
import graph
import rag_core
import responsible
import tracing
import v2
from rag_core import RagError, corpus_names, retrieve
from tracing import Trace, compute_cost, compute_fingerprint, record_trace

RETRIEVERS = ("dense", "bm25", "graph")
CANDIDATES = 20
RRF_K = 60


def hybrid_search(
    question: str,
    docs: list[str] | None = None,
    top_k: int = 5,
    retrievers: tuple[str, ...] | list[str] = ("dense",),
    candidates: int = CANDIDATES,
    rrf_k: int = RRF_K,
    weights: dict[str, float] | None = None,
    trace: Trace | None = None,
) -> dict[str, Any]:
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
        span_cm = trace.span(f"retrieve.{name}") if trace else None
        span = span_cm.__enter__() if span_cm else None
        started = time.time()
        try:
            if name == "dense":
                got = retrieve(question, docs, candidates)
            elif name == "bm25":
                got = bm25_index.search(question, docs, candidates, all_chunks)
            else:
                got, graph_path, note = graph.search(question, docs, candidates, by_id)
                if note:
                    notes.append(note)
            lists[name] = got
            dur_ms = round((time.time() - started) * 1000)
            timings[name] = dur_ms
            if span:
                span.set_attribute("candidates_count", len(got))
                span.set_attribute("candidate_ids", [c["id"] for c in got[:10]])
                if name == "dense":
                    span.set_attribute("distances", [round(c.get("distance", 0.0), 3) for c in got[:5]])
                span.finish("ok")
        except Exception as e:
            if span:
                span.set_attribute("error", str(e))
                span.finish("error")
            raise
        finally:
            if span_cm and trace:
                trace.current_span_id = span.parent_span_id if span else None

    # Reciprocal Rank Fusion
    span_cm = trace.span("fuse.rrf") if trace else None
    span = span_cm.__enter__() if span_cm else None
    try:
        ranked = {name: [c["id"] for c in got] for name, got in lists.items()}
        fused = fusion.rrf(ranked, k=rrf_k, weights=weights)
        dense_distance = {c["id"]: c.get("distance") for c in lists.get("dense", [])}
        chunks = []
        for f in fused[:top_k]:
            base = by_id.get(f["id"])
            if base is None:
                continue
            chunks.append(
                {**base, "distance": dense_distance.get(f["id"]), "rrf_score": f["score"], "found_by": f["found_by"]}
            )
        if span:
            span.set_attribute("top_k", top_k)
            span.set_attribute("rrf_k", rrf_k)
            span.set_attribute("fused_count", len(chunks))
            span.set_attribute("fused_chunk_ids", [c["id"] for c in chunks])
            span.finish("ok")
    except Exception as e:
        if span:
            span.set_attribute("error", str(e))
            span.finish("error")
        raise
    finally:
        if span_cm and trace:
            trace.current_span_id = span.parent_span_id if span else None

    return {
        "chunks": chunks,
        "per_retriever": ranked,
        "graph_path": graph_path,
        "notes": notes,
        "timings_ms": timings,
        "retrievers": retrievers,
    }


def ask_hybrid(
    question: str,
    docs: list[str] | None = None,
    top_k: int = 5,
    retrievers: tuple[str, ...] | list[str] = ("dense",),
    candidates: int = CANDIDATES,
    rrf_k: int = RRF_K,
    session_id: str | None = None,
    with_suggestions: bool = True,
) -> dict[str, Any]:
    if "chunks" not in inspect.signature(rag_core.ask).parameters:
        raise RagError(500, "patch_missing", "rag_core.ask has no 'chunks' parameter. Use the rag_core.py from the V3 pack.")

    started = time.time()
    docs = docs or corpus_names()

    # Create root trace for this question request
    fingerprint = compute_fingerprint(
        {"top_k": top_k, "retrievers": list(retrievers), "candidates": candidates, "rrf_k": rrf_k}
    )
    trace = Trace(
        name="request.ask",
        fingerprint=fingerprint,
        attributes={
            "retrievers": list(retrievers),
            "top_k": top_k,
            "docs": docs,
            "session_id": session_id,
        },
    )

    # 1. Responsible AI: PII Redaction Guardrail
    redacted_question, pii_meta = responsible.redact_pii(question)
    if pii_meta.get("pii_detected"):
        with trace.span("guardrail.pii") as s:
            s.set_attribute("redactions", pii_meta.get("details", {}))
            s.set_attribute("total_redactions", pii_meta.get("total_redactions", 0))

    # 2. Conversation Mode: Multi-turn Query Rewriting
    standalone_query = redacted_question
    session = None
    if session_id:
        session = conversation.get_or_create_session(session_id)
        standalone_query, _, _ = conversation.rewrite_query(session, redacted_question, trace=trace)

    # 3. Hybrid Search (Dense, BM25, Graph + RRF)
    result = hybrid_search(
        question=standalone_query,
        docs=docs,
        top_k=top_k,
        retrievers=retrievers,
        candidates=candidates,
        rrf_k=rrf_k,
        trace=trace,
    )

    if not result["chunks"]:
        trace.finish(outcome="abstained", status="error")
        record_trace(trace)
        raise RagError(409, "no_index", "No chunk was found. Check the selected documents and retrievers.")

    # 4. LLM Generation
    gen_cm = trace.span("generate")
    gen_span = gen_cm.__enter__()
    try:
        out = rag_core.ask(standalone_query, docs, top_k, chunks=result["chunks"])
        gen_span.set_attribute("model", rag_core.LLM_MODEL)
        gen_span.set_attribute("status", out.get("status"))
        if out.get("usage"):
            tin = out["usage"].get("input", 0)
            tout = out["usage"].get("output", 0)
            gen_span.set_attribute("tokens_in", tin)
            gen_span.set_attribute("tokens_out", tout)
            gen_span.set_attribute("cost_usd", compute_cost(tin, tout))
        gen_span.set_attribute("references", out.get("references", []))
        gen_span.finish("ok")
    except Exception as e:
        gen_span.set_attribute("error", str(e))
        gen_span.finish("error")
        trace.finish(outcome="error", status="error")
        record_trace(trace)
        raise
    finally:
        trace.current_span_id = gen_span.parent_span_id

    # 5. Conversation Mode: Suggested Next Questions
    suggestions = []
    if with_suggestions and out.get("status") == "answered":
        suggestions = conversation.suggest_next_questions(standalone_query, out.get("answer", ""), trace=trace)

    # 6. Finalize session and trace
    outcome = "answered" if out.get("status") == "answered" else "abstained"
    trace.finish(outcome=outcome, status="ok")
    record_trace(trace)

    if session:
        session.add_turn(
            user_question=question,
            rewritten_query=standalone_query,
            answer=out.get("answer", ""),
            references=out.get("references", []),
            trace_id=trace.trace_id,
        )

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
    out["trace_id"] = trace.trace_id
    out["trace"] = trace.to_dict()
    out["suggestions"] = suggestions
    out["session_id"] = session.session_id if session else None
    out["rewritten_query"] = standalone_query if standalone_query != question else None
    out["pii_detected"] = pii_meta.get("pii_detected", False)

    return out
