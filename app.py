import logging
from typing import Annotated, Literal

from flask import Flask, jsonify, render_template, request
from pydantic import BaseModel, Field, StringConstraints, ValidationError
from werkzeug.exceptions import HTTPException

import graph
import hybrid
import hybrid_eval
import v2
import words
from rag_core import (
    EMBEDDING_MODEL,
    RagError,
    Settings,
    add_doc,
    list_docs,
    load_settings,
    rebuild_all,
    remove_doc,
)

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


class DocRequest(BaseModel):
    name: str
    content: str


Retriever = Literal["dense", "bm25", "graph"]


class AskRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    docs: list[str] = Field(min_length=1)
    top_k: int = Field(ge=1, le=20)
    retrievers: list[Retriever] = Field(default=["dense"], min_length=1)


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


@app.get("/")
def index():
    return render_template("index.html")


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


@app.get("/api/docs")
def docs_list():
    return jsonify({"docs": list_docs(), "settings": load_settings().model_dump(), "model": EMBEDDING_MODEL})


@app.post("/api/docs")
def docs_add():
    req = validate(DocRequest, body_or_400())
    chunks = add_doc(req.name, req.content)
    return jsonify({"name": req.name, "chunks": chunks})


@app.delete("/api/docs/<name>")
def docs_delete(name):
    remove_doc(name)
    return jsonify({"deleted": name})


@app.post("/api/reindex")
def reindex():
    settings = validate(Settings, body_or_400())
    return jsonify({"chunks": rebuild_all(settings)})


@app.post("/api/ask")
def ask_route():
    req = validate(AskRequest, body_or_400())
    return jsonify(hybrid.ask_hybrid(req.question, req.docs, req.top_k, req.retrievers))


@app.post("/api/search")
def search_route():
    req = validate(SearchRequest, body_or_400())
    res = hybrid.hybrid_search(req.question, req.docs, req.top_k, req.retrievers)
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
