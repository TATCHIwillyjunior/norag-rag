import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import evalkit
import words
from rag_core import (
    BASE_DIR,
    CORPUS_DIR,
    EMBEDDING_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    RagError,
    Settings,
    ask,
    corpus_names,
    get_collection,
    load_settings,
    rebuild_all,
    retrieve,
)

MANIFEST = BASE_DIR / "eval" / "corpora.json"
RESULTS_DIR = BASE_DIR / "eval" / "results"
MAX_K = max(evalkit.KS)


def load_manifest():
    if not MANIFEST.exists():
        raise RagError(409, "no_manifest", "eval/corpora.json is missing.")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def corpus_spec(name):
    manifest = load_manifest()
    if name not in manifest:
        raise RagError(400, "invalid_request", f"Unknown corpus: {name}.")
    return manifest[name]


def corpora_info():
    out = {}
    for name, spec in load_manifest().items():
        path = BASE_DIR / spec["questions"]
        count = len(evalkit.load_questions(path)) if path.exists() else 0
        out[name] = {"docs": spec["docs"], "questions": count}
    return out


def all_questions():
    out = []
    for name, spec in load_manifest().items():
        path = BASE_DIR / spec["questions"]
        if not path.exists():
            continue
        for q in evalkit.load_questions(path):
            out.append({"corpus": name, "id": q["id"], "question": q["question"]})
    return out


def all_chunks(corpus=None):
    got = get_collection().get(include=["documents", "metadatas"])
    chunks = [
        {"id": i, "doc": m["doc"], "position": m["position"], "start": m["start"], "end": m["end"], "text": t}
        for i, m, t in zip(got["ids"], got["metadatas"], got["documents"])
    ]
    if corpus:
        docs = set(corpus_spec(corpus)["docs"])
        chunks = [c for c in chunks if c["doc"] in docs]
    chunks.sort(key=lambda c: (c["doc"], c["position"]))
    return chunks


def words_context(corpus):
    chunks = all_chunks(corpus or None)
    return words.build_index(chunks), {c["id"]: c for c in chunks}, chunks


def search(question, docs, top_k):
    docs = docs or corpus_names()
    return retrieve(question, docs, top_k)


def _read_doc(name):
    return (CORPUS_DIR / name).read_text(encoding="utf-8", errors="replace")


def _config_name(settings):
    return f"c{settings.chunk_size}-o{settings.overlap_percent}"


def run_eval(corpus, with_answers=False, answer_k=5):
    spec = corpus_spec(corpus)
    path = BASE_DIR / spec["questions"]
    if not path.exists():
        raise RagError(409, "no_questions", f"{spec['questions']} does not exist yet.")
    questions = evalkit.load_questions(path)
    docs = spec["docs"]
    present = set(corpus_names())
    missing = [d for d in docs if d not in present]
    if missing:
        raise RagError(409, "missing_docs", "These documents are not in the app: " + ", ".join(missing))

    problems = evalkit.check_evidence(questions, _read_doc)
    settings = load_settings()
    rows = []
    for q in questions:
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answerable": q["answerable"],
        }
        chunks = retrieve(q["question"], docs, MAX_K)
        row["retrieved"] = [c["id"] for c in chunks]
        row["top_distance"] = chunks[0]["distance"] if chunks else None
        if q["answerable"]:
            ranks = evalkit.evidence_ranks(q, chunks)
            hit, recall, first = evalkit.score(ranks)
            row["evidence_ranks"] = ranks
            row["hit"] = hit
            row["recall"] = recall
            row["first_rank"] = first
        if with_answers:
            try:
                out = ask(q["question"], docs, answer_k)
                row["answer"] = {
                    "status": out["status"],
                    "text": out["answer"],
                    "references": out["references"],
                    "warnings": out["warnings"],
                    "chunks": [{"id": c["id"], "text": c["text"]} for c in out["chunks"]],
                }
            except RagError as e:
                row["answer"] = {"status": "error", "text": e.message, "references": [], "warnings": [], "chunks": []}
            row["expected_answer"] = q.get("expected_answer")
            row["grade"] = None
            row["failure_stage"] = None
            row["grading_notes"] = ""
        rows.append(row)

    result = {
        "corpus": corpus,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_size": settings.chunk_size,
        "overlap_percent": settings.overlap_percent,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL if with_answers else None,
        "answer_top_k": answer_k if with_answers else None,
        "evidence_problems": problems,
        "summary": evalkit.summarize(rows),
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{corpus}-{_config_name(settings)}"
    if with_answers:
        name += f"-answers-k{answer_k}"
    file = RESULTS_DIR / f"{name}.json"
    file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["file"] = str(file.relative_to(BASE_DIR))
    return result


def no_context_test(corpus):
    from openai import OpenAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RagError(500, "not_configured", "GOOGLE_API_KEY is missing in the .env file.")
    spec = corpus_spec(corpus)
    path = BASE_DIR / spec["questions"]
    if not path.exists():
        raise RagError(409, "no_questions", f"{spec['questions']} does not exist yet.")
    client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key, timeout=30, max_retries=1)
    rows = []
    for q in evalkit.load_questions(path):
        try:
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Answer briefly. If you do not know, say so."},
                    {"role": "user", "content": q["question"]},
                ],
            )
            answer = completion.choices[0].message.content or ""
        except Exception as e:
            answer = f"ERROR: {type(e).__name__}"
        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "answerable": q["answerable"],
                "expected_answer": q.get("expected_answer"),
                "model_answer": answer,
                "model_knew_it": None,
            }
        )
        time.sleep(0.3)
    result = {
        "corpus": corpus,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_model": LLM_MODEL,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file = RESULTS_DIR / f"no-context-{corpus}.json"
    file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    result["file"] = str(file.relative_to(BASE_DIR))
    return result


def reconfigure(chunk_size, overlap_percent):
    return rebuild_all(Settings(chunk_size=chunk_size, overlap_percent=overlap_percent))
