"""Graph assembly and entity resolution.

Entity resolution, in two steps:
1. Normalization: names with the same normalized key AND the same type become one node
   ("Postroom", "postroom", "the postroom service" -> Service:postroom).
   Same key but different types are NOT merged (logged as "not_merged_type_conflict"),
   because "Messaging" the team and "messaging" the feature are not the same thing.
2. Explicit aliases: a SAME_AS triple merges its two nodes (union-find), whatever their type.
   The text said so, and the evidence chunk is kept on the merged node (merge_chunk_ids).

Every node keeps all the original names it was seen with (aliases) and every chunk it came from.
Every edge keeps its chunk ID and its evidence quote.
"""
import re
import unicodedata
from collections import Counter, defaultdict

from .schema import EntityType, RelationType

_GENERIC_SUFFIX = {
    EntityType.SERVICE.value: (" service",),
    EntityType.TEAM.value: (" team",),
}


def normalize(name, etype=None):
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.replace("’", "'").replace("**", "")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^\w\s\-#@.]", " ", s)
    s = " ".join(s.split()).strip(" .")
    s = re.sub(r"^(the|a|an) ", "", s)
    for suffix in _GENERIC_SUFFIX.get(etype, ()):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
    return s


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_graph(triples_by_chunk):
    """triples_by_chunk: list of (chunk_id, [Triple]). Returns (graph dict, merges list)."""
    mentions = defaultdict(Counter)        # node_key -> Counter(surface name)
    node_chunks = defaultdict(set)
    node_type = {}
    raw_edges = []
    same_as = []

    def key_of(ref):
        t = ref.type.value
        return f"{t}:{normalize(ref.name, t)}"

    for chunk_id, triples in triples_by_chunk:
        for tr in triples:
            ks, ko = key_of(tr.subject), key_of(tr.object)
            for k, ref in ((ks, tr.subject), (ko, tr.object)):
                mentions[k][ref.name] += 1
                node_chunks[k].add(chunk_id)
                node_type[k] = ref.type.value
            if tr.relation == RelationType.SAME_AS:
                same_as.append((ks, ko, chunk_id, tr.evidence))
            else:
                raw_edges.append((ks, tr.relation.value, ko, chunk_id, tr.evidence))

    merges = []
    # Step 1 report: several surface names collapsed into one node by normalization.
    for k, names in mentions.items():
        if len(names) > 1:
            merges.append({"kind": "normalization", "node": k, "names": sorted(names)})
    # Same normalized name, different types: kept apart on purpose.
    by_plain = defaultdict(list)
    for k in mentions:
        by_plain[k.split(":", 1)[1]].append(k)
    for plain, keys in by_plain.items():
        if len(keys) > 1:
            merges.append({"kind": "not_merged_type_conflict", "name": plain, "nodes": sorted(keys)})

    # Step 2: explicit aliases.
    uf = _UnionFind()
    for k in mentions:
        uf.find(k)
    merge_chunks = defaultdict(set)
    for a, b, chunk_id, evidence in same_as:
        uf.union(a, b)
        merges.append({"kind": "same_as", "merged": [a, b], "chunk_id": chunk_id, "evidence": evidence})
        merge_chunks[a].add(chunk_id)
        merge_chunks[b].add(chunk_id)

    groups = defaultdict(list)
    for k in mentions:
        groups[uf.find(k)].append(k)

    nodes = {}
    rep = {}
    for root, members in groups.items():
        names = Counter()
        chunks, mchunks = set(), set()
        for m in members:
            names.update(mentions[m])
            chunks |= node_chunks[m]
            mchunks |= merge_chunks[m]
        # Canonical: most mentioned, then fewest words (a proper name beats a description), then A-Z.
        best = sorted(members, key=lambda m: (-sum(mentions[m].values()),
                                              len(m.split()), m))[0]
        display = mentions[best].most_common(1)[0][0]
        node_id = best
        for m in members:
            rep[m] = node_id
        nodes[node_id] = {"id": node_id, "name": display, "type": node_type[best],
                          "aliases": sorted(names), "merged_keys": sorted(members),
                          "chunk_ids": sorted(chunks), "merge_chunk_ids": sorted(mchunks)}

    edges, seen = [], set()
    for s, rel, o, chunk_id, evidence in raw_edges:
        s, o = rep[s], rep[o]
        if s == o:
            continue
        sig = (s, rel, o, chunk_id)
        if sig in seen:
            continue
        seen.add(sig)
        edges.append({"id": f"e{len(edges)}", "source": s, "relation": rel, "target": o,
                      "chunk_id": chunk_id, "evidence": evidence})

    graph = {"nodes": sorted(nodes.values(), key=lambda n: n["id"]), "edges": edges}
    return graph, merges
