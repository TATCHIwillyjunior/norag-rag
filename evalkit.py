import json
from pathlib import Path

KS = (1, 3, 5, 10)
TYPES = ("exact_identifier", "split_answer", "vocabulary", "not_in_corpus", "versions", "two_hop", "other")


def norm(s):
    return " ".join(s.split()).lower()


def load_questions(path):
    questions = []
    seen = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {number}: {e}")
        for key in ("id", "question", "type", "answerable", "evidence"):
            if key not in q:
                raise ValueError(f"{path} line {number}: missing '{key}'")
        if q["id"] in seen:
            raise ValueError(f"{path} line {number}: duplicate id {q['id']}")
        if q["answerable"] and not q["evidence"]:
            raise ValueError(f"{path} line {number}: {q['id']} is answerable but has no evidence")
        seen.add(q["id"])
        questions.append(q)
    return questions


def check_evidence(questions, read_doc):
    problems = []
    for q in questions:
        for ev in q["evidence"]:
            try:
                text = read_doc(ev["source"])
            except FileNotFoundError:
                problems.append(f"{q['id']}: source {ev['source']} is not in the corpus folder")
                continue
            if norm(ev["quote"]) not in norm(text):
                problems.append(f"{q['id']}: quote not found in {ev['source']}: {ev['quote'][:60]}")
    return problems


def evidence_ranks(question, chunks):
    ranks = []
    for ev in question["evidence"]:
        needle = norm(ev["quote"])
        rank = None
        for i, c in enumerate(chunks, 1):
            if c["doc"] == ev["source"] and needle in norm(c["text"]):
                rank = i
                break
        ranks.append(rank)
    return ranks


def score(ranks):
    hit = {}
    recall = {}
    for k in KS:
        found = [r for r in ranks if r is not None and r <= k]
        hit[f"hit@{k}"] = 1 if found else 0
        recall[f"recall@{k}"] = len(found) / len(ranks)
    present = [r for r in ranks if r is not None]
    first = min(present) if present else None
    return hit, recall, first


def _aggregate(rows):
    n = len(rows)
    if n == 0:
        return None
    out = {"n": n}
    for k in KS:
        out[f"hit@{k}"] = sum(r["hit"][f"hit@{k}"] for r in rows) / n
    for k in KS:
        out[f"recall@{k}"] = sum(r["recall"][f"recall@{k}"] for r in rows) / n
    out["mrr"] = sum(1 / r["first_rank"] for r in rows if r["first_rank"]) / n
    return out


def summarize(rows):
    answerable = [r for r in rows if r["answerable"]]
    types = sorted({r["type"] for r in answerable})
    return {
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(rows) - len(answerable),
        "overall": _aggregate(answerable),
        "per_type": {t: _aggregate([r for r in answerable if r["type"] == t]) for t in types},
    }
