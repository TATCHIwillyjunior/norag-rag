import hashlib
import json
import math
from collections import Counter

import words
from rag_core import BASE_DIR

INDEX_FILE = BASE_DIR / "data" / "bm25_index.json"
K1 = 1.5
B = 0.75

_cache = {"fingerprint": None, "index": None}


def fingerprint(chunks):
    h = hashlib.sha1()
    for c in sorted(chunks, key=lambda c: c["id"]):
        h.update(c["id"].encode("utf-8"))
        h.update(b"\0")
        h.update(c["text"].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def build(chunks):
    entries = {}
    for c in chunks:
        tokens = words.tokenize(c["text"])
        entries[c["id"]] = {"doc": c["doc"], "len": len(tokens), "tf": dict(Counter(tokens))}
    return {
        "k1": K1,
        "b": B,
        "tokenizer": "lowercase, split on non letters/digits",
        "fingerprint": fingerprint(chunks),
        "n_chunks": len(entries),
        "entries": entries,
    }


def get_index(chunks):
    fp = fingerprint(chunks)
    if _cache["fingerprint"] == fp:
        return _cache["index"], "memory"
    if INDEX_FILE.exists():
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fp:
                _cache["fingerprint"], _cache["index"] = fp, data
                return data, "file"
        except (OSError, ValueError):
            pass
    index = build(chunks)
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    _cache["fingerprint"], _cache["index"] = fp, index
    return index, "rebuilt"


def search(question, docs, limit, chunks):
    chunks = list(chunks)
    by_id = {c["id"]: c for c in chunks}
    index, _source = get_index(chunks)
    entries = index["entries"]
    pool = [cid for cid, e in entries.items() if e["doc"] in docs and cid in by_id]
    terms = list(dict.fromkeys(words.tokenize(question)))
    if not pool or not terms:
        return []
    n = len(pool)
    avgdl = sum(entries[cid]["len"] for cid in pool) / n or 1.0
    df = {t: sum(1 for cid in pool if t in entries[cid]["tf"]) for t in terms}
    scores = {}
    for cid in pool:
        e = entries[cid]
        total = 0.0
        for t in terms:
            f = e["tf"].get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            total += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * e["len"] / avgdl))
        if total > 0:
            scores[cid] = total
    ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    return [{**by_id[cid], "bm25_score": scores[cid]} for cid in ranked]
