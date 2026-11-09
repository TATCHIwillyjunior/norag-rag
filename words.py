import re
from collections import Counter

TOKEN = re.compile(r"\w+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def build_index(chunks):
    index = {}
    for c in chunks:
        for word, n in Counter(tokenize(c["text"])).items():
            index.setdefault(word, {})[c["id"]] = n
    return index


def stats(index, chunks):
    occurrences = sum(sum(per.values()) for per in index.values())
    return {"chunks": len(chunks), "vocabulary": len(index), "occurrences": occurrences}


def vocabulary(index, contains="", limit=200):
    needle = contains.strip().lower()
    rows = []
    for word, per in index.items():
        if needle and needle not in word:
            continue
        rows.append({"word": word, "occurrences": sum(per.values()), "chunks": len(per)})
    rows.sort(key=lambda r: (-r["occurrences"], r["word"]))
    return {"total_matching": len(rows), "rows": rows[:limit]}


def word_chunks(index, by_id, word, limit=100):
    word = word.strip().lower()
    tokens = tokenize(word)
    note = ""
    if not tokens:
        note = "No letters or digits in this input."
    elif tokens != [word]:
        note = f"'{word}' is read as {len(tokens)} words: {', '.join(tokens)}. Look them up one by one."
    per = index.get(word, {})
    rows = [
        {
            "id": cid,
            "doc": by_id[cid]["doc"],
            "position": by_id[cid]["position"],
            "count": n,
            "text": by_id[cid]["text"],
        }
        for cid, n in per.items()
    ]
    rows.sort(key=lambda r: (-r["count"], r["doc"], r["position"]))
    return {
        "word": word,
        "note": note,
        "exists": bool(per),
        "occurrences": sum(per.values()),
        "n_chunks": len(rows),
        "chunks": rows[:limit],
    }


def question_words(index, question):
    seen = set()
    rows = []
    for w in tokenize(question):
        if w in seen:
            continue
        seen.add(w)
        per = index.get(w, {})
        top = sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        rows.append(
            {
                "word": w,
                "exists": bool(per),
                "occurrences": sum(per.values()),
                "n_chunks": len(per),
                "top_chunks": [{"id": cid, "count": n} for cid, n in top],
            }
        )
    return {"rows": rows}
