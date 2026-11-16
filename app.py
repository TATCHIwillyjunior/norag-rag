import logging
import time
from typing import Annotated, Literal

from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, Field, StringConstraints, ValidationError
from werkzeug.exceptions import HTTPException

import conversation
import explorer
import graph
import hybrid
import hybrid_eval
import responsible
import tracing
import v2
import words
from rag_core import (
    EMBEDDING_MODEL,
    RagError,
    Settings,
    add_doc,
    corpus_names,
    index_doc,
    list_docs,
    load_settings,
    rebuild_all,
    remove_doc,
    reset_collection,
    save_settings,
)
from tracing import Trace, record_trace

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


class DocRequest(BaseModel):
    name: str
    content: str


class QuarantineApproveRequest(BaseModel):
    name: str


Retriever = Literal["dense", "bm25", "graph"]


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    docs: list[str] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    retrievers: list[Retriever] = Field(default=["dense"], min_length=1)
    session_id: str | None = None


class SearchRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    docs: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    retrievers: list[Retriever] = Field(default=["dense"], min_length=1)


class EvalRequest(BaseModel):
    corpus: str
    answers: bool = False
    retrievers: list[Retriever] = Field(default=["dense"], min_length=1)


class WordsQuestionRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    corpus: str = ""


def body_or_400() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise RagError(400, "invalid_request", "Invalid JSON body.")
    return data


def validate(model, data):
    try:
        return model.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        field = ".".join(str(x) for x in first["loc"])
        raise RagError(400, "invalid_request", f"{field}: {first['msg']}")


@app.errorhandler(RagError)
def handle_rag_error(e):
    return jsonify({"error": e.code, "message": e.message}), e.status


@app.errorhandler(413)
def handle_too_large(e):
    return jsonify({"error": "invalid_request", "message": "The request is too large (max 5 MB)."}), 413


@app.errorhandler(Exception)
def handle_unexpected(e):
    if isinstance(e, HTTPException):
        return e
    app.logger.exception("unexpected error")
    return jsonify({"error": "internal_error", "message": "Internal error."}), 500


# ---------------- PAGES ----------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/explorer")
def explorer_page():
    return render_template("explorer.html")


@app.get("/metrics")
def metrics_page():
    return render_template("metrics.html")


@app.get("/quarantine")
def quarantine_page():
    return render_template("quarantine.html")


@app.get("/eval")
def eval_page():
    return render_template("eval.html")


@app.get("/words")
def words_page():
    return render_template("words.html")


@app.get("/bm25")
def bm25_page():
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>BM25</title></head>'
        '<body style="font-family:sans-serif;margin:40px">'
        '<p style="font-size:64px">Watch your sources!</p></body></html>'
    )


@app.get("/about")
def about():
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>About</title></head>'
        '<body style="font-family:sans-serif;margin:40px">'
        '<p style="font-size:48px">Denis Amselemovitch is an intriguing character who likes to make '
        "impromptu appearances in incongruous places.</p></body></html>"
    )


# ---------------- OBSERVABILITY APIs ----------------

@app.get("/api/traces")
def get_traces_list():
    limit = int(request.args.get("limit", 50))
    return jsonify({"traces": tracing.list_traces(limit)})


@app.get("/api/traces/<trace_id>")
def get_trace_by_id(trace_id: str):
    t = tracing.get_trace(trace_id)
    if not t:
        raise RagError(404, "not_found", f"Trace {trace_id} not found.")
    return jsonify(t)


@app.get("/api/metrics")
def get_metrics_api():
    return jsonify(tracing.get_metrics())


# ---------------- CONCEPT MAP EXPLORER APIs ----------------

@app.get("/api/graph/explorer")
def graph_explorer():
    return jsonify(explorer.get_explorer_graph())


@app.get("/api/graph/node")
def graph_node_details():
    node_id = request.args.get("id") or request.args.get("name", "")
    if not node_id:
        raise RagError(400, "invalid_request", "Missing 'id' parameter.")
    return jsonify(explorer.get_node_details(node_id))


# ---------------- CONVERSATION APIs ----------------

@app.delete("/api/chat/<session_id>")
def delete_chat_session(session_id: str):
    success = conversation.clear_session(session_id)
    return jsonify({"session_id": session_id, "cleared": success})


# ---------------- RESPONSIBLE AI & DOCUMENT MANAGEMENT APIs ----------------

@app.get("/api/docs")
def docs_list():
    return jsonify({"docs": list_docs(), "settings": load_settings().model_dump(), "model": EMBEDDING_MODEL})


@app.post("/api/docs")
def docs_add():
    req = validate(DocRequest, body_or_400())

    # Step 1: Responsible AI Prompt Injection & Poisoning Scan
    scan = responsible.scan_injection(req.content, req.name)
    if scan.get("is_suspicious"):
        meta = responsible.quarantine_document(req.name, req.content, scan)
        return (
            jsonify(
                {
                    "name": req.name,
                    "quarantined": True,
                    "risk_score": scan["risk_score"],
                    "reasons": scan["reasons"],
                    "message": "File placed in Quarantine due to suspicious prompt injection or poisoning content.",
                }
            ),
            202,
        )

    # Step 2: Safe document indexing
    t0 = time.time()
    chunks = add_doc(req.name, req.content)
    dur_ms = round((time.time() - t0) * 1000)

    # Trace single doc indexing
    idx_trace = Trace(name="indexing.run", attributes={"doc": req.name, "chunks": chunks, "duration_ms": dur_ms})
    with idx_trace.span("index.doc") as s:
        s.set_attribute("doc", req.name)
        s.set_attribute("chunks", chunks)
        s.set_attribute("embedding_time_ms", dur_ms)
    idx_trace.finish(outcome="indexed", status="ok")
    record_trace(idx_trace)

    return jsonify({"name": req.name, "chunks": chunks, "quarantined": False})


@app.get("/api/quarantine")
def list_quarantine_docs():
    return jsonify({"quarantined": responsible.list_quarantined()})


@app.get("/api/quarantine/<name>")
def get_quarantine_doc(name: str):
    try:
        return jsonify(responsible.get_quarantined_content(name))
    except FileNotFoundError as e:
        raise RagError(404, "not_found", str(e))


@app.post("/api/quarantine/approve")
def approve_quarantine_doc():
    req = validate(QuarantineApproveRequest, body_or_400())
    try:
        res = responsible.approve_quarantine(req.name)
        return jsonify(res)
    except FileNotFoundError as e:
        raise RagError(404, "not_found", str(e))


@app.delete("/api/quarantine/<name>")
def delete_quarantine_doc(name: str):
    res = responsible.delete_quarantine(name)
    return jsonify(res)


@app.delete("/api/docs/<name>")
def docs_delete(name: str):
    # GDPR Article 17: erase from every store and return the verification report.
    report = responsible.purge_document(name)
    return jsonify({"deleted": name, "purge": report})


@app.post("/api/reindex")
def reindex():
    settings = validate(Settings, body_or_400())
    t0 = time.time()
    idx_trace = Trace(
        name="indexing.run",
        attributes={"chunk_size": settings.chunk_size, "overlap": settings.overlap_percent},
    )

    save_settings(settings)
    reset_collection()
    total_chunks = 0
    docs = corpus_names()

    # Track each document indexed as a span with duration and chunk count
    for doc_name in docs:
        t_doc = time.time()
        with idx_trace.span("index.doc") as s:
            cnt = index_doc(doc_name)
            total_chunks += cnt
            s.set_attribute("doc", doc_name)
            s.set_attribute("chunks", cnt)
            s.set_attribute("embedding_time_ms", round((time.time() - t_doc) * 1000, 2))

    # Track concept graph extraction if present
    gst = graph.stats()
    if gst and gst.get("build_cost_total"):
        with idx_trace.span("index.graph") as gs:
            cost_info = gst.get("build_cost_total", {})
            gs.set_attribute("entities", gst.get("entities", 0))
            gs.set_attribute("relations", gst.get("relations", 0))
            gs.set_attribute("graph_calls", cost_info.get("calls", 0))
            gs.set_attribute("tokens_in", cost_info.get("in", 0))
            gs.set_attribute("tokens_out", cost_info.get("out", 0))
            gs.set_attribute("cost_usd", tracing.compute_cost(cost_info.get("in", 0), cost_info.get("out", 0)))

    idx_trace.attributes["total_chunks"] = total_chunks
    idx_trace.attributes["docs_count"] = len(docs)
    idx_trace.finish(outcome="indexed", status="ok")
    record_trace(idx_trace)

    return jsonify({"chunks": total_chunks, "trace_id": idx_trace.trace_id})


# ---------------- QUERY & ASK ROUTE ----------------

@app.post("/api/ask")
def ask_route():
    req = validate(AskRequest, body_or_400())
    return jsonify(
        hybrid.ask_hybrid(
            question=req.question,
            docs=req.docs,
            top_k=req.top_k,
            retrievers=req.retrievers,
            session_id=req.session_id,
        )
    )


@app.post("/api/search")
def search_route():
    req = validate(SearchRequest, body_or_400())
    res = hybrid.hybrid_search(
        question=req.question,
        docs=req.docs,
        top_k=req.top_k,
        retrievers=req.retrievers,
    )
    return jsonify(res)


@app.get("/api/corpora")
def corpora_route():
    return jsonify({"corpora": v2.corpora_info(), "settings": load_settings().model_dump()})


@app.post("/api/eval")
def eval_route():
    req = validate(EvalRequest, body_or_400())
    return jsonify(hybrid_eval.run_eval(req.corpus, req.retrievers, with_answers=req.answers))


@app.get("/api/graph/stats")
def graph_stats():
    return jsonify({"stats": graph.stats()})


@app.get("/api/questions")
def questions_route():
    return jsonify({"questions": v2.all_questions()})


@app.get("/api/words/vocab")
def words_vocab():
    corpus = request.args.get("corpus", "")
    contains = request.args.get("q", "")
    try:
        limit = max(1, min(int(request.args.get("limit", "200")), 1000))
    except ValueError:
        raise RagError(400, "invalid_request", "limit must be a number.")
    index, by_id, chunks = v2.words_context(corpus)
    out = words.vocabulary(index, contains, limit)
    out["stats"] = words.stats(index, chunks)
    return jsonify(out)


@app.get("/api/words/word")
def words_word():
    corpus = request.args.get("corpus", "")
    word = request.args.get("w", "").strip()
    if not word:
        raise RagError(400, "invalid_request", "Type a word.")
    index, by_id, chunks = v2.words_context(corpus)
    return jsonify(words.word_chunks(index, by_id, word))


@app.post("/api/words/question")
def words_question():
    req = validate(WordsQuestionRequest, body_or_400())
    index, by_id, chunks = v2.words_context(req.corpus)
    return jsonify(words.question_words(index, req.question))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
