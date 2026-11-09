def rrf(ranked, k=60, weights=None):
    scores = {}
    found = {}
    for name, ids in ranked.items():
        weight = 1.0 if not weights else weights.get(name, 1.0)
        for rank, chunk_id in enumerate(ids, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            found.setdefault(chunk_id, {})[name] = rank
    order = sorted(scores, key=lambda cid: (-scores[cid], min(found[cid].values()), cid))
    return [{"id": cid, "score": scores[cid], "found_by": found[cid]} for cid in order]
