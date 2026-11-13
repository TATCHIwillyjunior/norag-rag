import json
import time
from datetime import datetime, timezone

import evalkit
import hybrid
import v2
from rag_core import BASE_DIR, CORPUS_DIR, EMBEDDING_MODEL, LLM_MODEL, RagError, corpus_names, load_settings

RESULTS_DIR = BASE_DIR / "eval" / "results"
MAX_K = max(evalkit.KS)


def _read_doc(name):
    return (CORPUS_DIR / name).read_text(encoding="utf-8", errors="replace")


def _pseudo_rows(rows, retriever):
    out = []
    for r in rows:
        if not r["answerable"]:
            continue
        ranks = r["retriever_ranks"][retriever]
        hit, recall, first = evalkit.score(ranks)
        out.append({"answerable": True, "type": r["type"], "hit": hit, "recall": recall, "first_rank": first})
    return out


def run_eval(corpus, retrievers, with_answers=False, answer_k=5, candidates=hybrid.CANDIDATES, rrf_k=hybrid.RRF_K):
    spec = v2.corpus_spec(corpus)
    path = BASE_DIR / spec["questions"]
    if not path.exists():
        raise RagError(409, "no_questions", f"{spec['questions']} does not exist yet.")
    questions = evalkit.load_questions(path)
    docs = spec["docs"]
    missing = [d for d in docs if d not in set(corpus_names())]
    if missing:
        raise RagError(409, "missing_docs", "These documents are not in the app: " + ", ".join(missing))
    retrievers = list(dict.fromkeys(retrievers))
    problems = evalkit.check_evidence(questions, _read_doc)
    settings = load_settings()
    by_id = {c["id"]: c for c in v2.all_chunks()}

    rows = []
    for q in questions:
        started = time.time()
        res = hybrid.hybrid_search(q["question"], docs, MAX_K, retrievers, candidates, rrf_k)
        latency = round((time.time() - started) * 1000)
        chunks = res["chunks"]
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answerable": q["answerable"],
            "retrieved": [c["id"] for c in chunks],
            "found_by": {c["id"]: c["found_by"] for c in chunks},
            "per_retriever": {n: ids[:MAX_K] for n, ids in res["per_retriever"].items()},
            "retrieval_latency_ms": latency,
            "retriever_timings_ms": res["timings_ms"],
            "notes": res["notes"],
        }
        if q["answerable"]:
            ranks = evalkit.evidence_ranks(q, chunks)
            hit, recall, first = evalkit.score(ranks)
            row.update({"evidence_ranks": ranks, "hit": hit, "recall": recall, "first_rank": first})
            row["retriever_ranks"] = {}
            for name, ids in res["per_retriever"].items():
                listed = [by_id[i] for i in ids[:MAX_K] if i in by_id]
                row["retriever_ranks"][name] = evalkit.evidence_ranks(q, listed)
            if "graph" in res["per_retriever"]:
                needles = [(ev["source"], evalkit.norm(ev["quote"])) for ev in q["evidence"]]

                def relevant(cid):
                    c = by_id.get(cid)
                    return bool(c) and any(c["doc"] == s and n in evalkit.norm(c["text"]) for s, n in needles)

                graph_ids = res["per_retriever"]["graph"]
                relevant_graph = [cid for cid in graph_ids if relevant(cid)]
                top5 = set(row["retrieved"][:5])
                row["graph_relevant_chunks"] = relevant_graph
                row["graph_relevant_in_fused_top5"] = [cid for cid in relevant_graph if cid in top5]
        if with_answers:
            try:
                out = hybrid.ask_hybrid(q["question"], docs, answer_k, retrievers, candidates, rrf_k)
                row["answer"] = {
                    "status": out["status"],
                    "text": out["answer"],
                    "references": out["references"],
                    "reference_details": out["reference_details"],
                    "warnings": out["warnings"],
                    "chunks": [{"id": c["id"], "text": c["text"]} for c in out["chunks"]],
                    "usage": out["usage"],
                    "latency_ms": out["total_ms"],
                    "llm_calls": 1,
                }
            except RagError as e:
                row["answer"] = {"status": "error", "text": e.message, "references": [], "reference_details": [], "warnings": [], "chunks": [], "usage": None, "latency_ms": 0, "llm_calls": 0}
            row["expected_answer"] = q.get("expected_answer")
            row["grade"] = None
            row["failure_stage"] = None
            row["grading_notes"] = ""
        rows.append(row)

    n = len(rows) or 1
    cost = {"avg_retrieval_ms": round(sum(r["retrieval_latency_ms"] for r in rows) / n), "retrieval_llm_calls_per_question": 0}
    if with_answers:
        answered = [r["answer"] for r in rows if r["answer"]["status"] != "error"]
        if answered:
            cost["avg_answer_ms"] = round(sum(a["latency_ms"] for a in answered) / len(answered))
            cost["avg_tokens_per_answer"] = round(sum((a["usage"] or {}).get("total", 0) for a in answered) / len(answered))
            cost["answer_llm_calls_per_question"] = 1

    result = {
        "corpus": corpus,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "retrievers": retrievers,
        "candidates": candidates,
        "rrf_k": rrf_k,
        "chunk_size": settings.chunk_size,
        "overlap_percent": settings.overlap_percent,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL if with_answers else None,
        "answer_top_k": answer_k if with_answers else None,
        "evidence_problems": problems,
        "summary": evalkit.summarize(rows),
        "per_retriever_summary": {name: evalkit.summarize(_pseudo_rows(rows, name)) for name in retrievers},
        "cost": cost,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{corpus}-c{settings.chunk_size}-o{settings.overlap_percent}-{'+'.join(retrievers)}-n{candidates}-rrf{rrf_k}"
    if with_answers:
        name += f"-answers-k{answer_k}"
    file = RESULTS_DIR / f"{name}.json"
    file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["file"] = str(file.relative_to(BASE_DIR))
    return result


def compare(file_a, file_b, k=5):
    a = json.loads(open(file_a, encoding="utf-8").read())
    b = json.loads(open(file_b, encoding="utf-8").read())
    rows_b = {r["id"]: r for r in b["rows"]}
    worse, better = [], []
    for r in a["rows"]:
        if not r["answerable"] or r["id"] not in rows_b:
            continue
        ra = r["recall"][f"recall@{k}"]
        rb = rows_b[r["id"]]["recall"][f"recall@{k}"]
        item = {"id": r["id"], "type": r["type"], "question": r["question"], f"A recall@{k}": ra, f"B recall@{k}": rb}
        if rb < ra:
            worse.append(item)
        elif rb > ra:
            better.append(item)
    return {"A": "+".join(a["retrievers"]), "B": "+".join(b["retrievers"]), "k": k, "worse_in_B": worse, "better_in_B": better}
