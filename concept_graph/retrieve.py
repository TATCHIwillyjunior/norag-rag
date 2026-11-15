"""Graph retriever: question -> entities -> paths (up to 2 hops) -> ranked source chunks.

Strategy (no LLM call at query time, so the graph adds no tokens per question):
1. Entity linking: a node matches if one of its names (normalized) appears in the normalized
   question as whole words. Names of 1 character, and very common one-word names, are ignored.
2. Traversal: breadth-first from the matched nodes, following edges in both directions,
   up to max_hops (default 2). An edge first reached at hop h gets weight 1 / h.
   A hub (more than max_degree edges, e.g. "Kestrel") reached during the walk is not expanded,
   otherwise one hub would pull half the corpus into the result.
   When the walk enters a node that was merged through SAME_AS, the chunk that states the alias
   is added with the same weight: it is the link between the two hops.
3. Chunk score = sum of the weights of the edges (and alias links) that cite the chunk.
   Ties are broken by chunk ID, so the ranking is deterministic.

Result shape (shared retriever interface):
  {"chunk_id", "text", "source", "score", "rank", "retriever": "graph", "path": [edges...]}
plus, from explain(): matched entities and the reason when nothing is returned.
"""
import json
import re
from collections import defaultdict

from .graph import normalize

_STOP_NAMES = {"it", "he", "she", "they", "we", "you", "team", "service", "user", "users",
               "customer", "customers", "support", "error", "plan", "workspace", "data"}


class GraphRetriever:
    name = "graph"

    def __init__(self, graph, chunks, max_hops=2, max_degree=20):
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.edges = graph["edges"]
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.max_hops = max_hops
        self.max_degree = max_degree  # hub nodes reached during the walk are not expanded further
        self.adj = defaultdict(list)
        for e in self.edges:
            self.adj[e["source"]].append((e, e["target"]))
            self.adj[e["target"]].append((e, e["source"]))
        self.alias_index = []
        for n in self.nodes.values():
            for alias in set(n["aliases"]) | {n["name"]}:
                a = normalize(alias, n["type"])
                if len(a) < 2 or a in _STOP_NAMES:
                    continue
                self.alias_index.append((a, n["id"]))

    @classmethod
    def load(cls, graph_path, chunks_path, **kw):
        with open(graph_path, encoding="utf-8") as f:
            graph = json.load(f)
        chunks = []
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return cls(graph, chunks, **kw)

    def link_entities(self, question):
        q = " " + normalize(question) + " "
        found = {}
        for alias, node_id in self.alias_index:
            if re.search(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", q):
                found[node_id] = max(found.get(node_id, ""), alias, key=len)
        return found  # node_id -> matched text

    def explain(self, question, top_k=10):
        seeds = self.link_entities(question)
        if not seeds:
            return {"entities": [], "results": [], "reason": "no_entity_match"}
        chunk_score = defaultdict(float)
        chunk_path = defaultdict(list)
        visited = {s: 0 for s in seeds}
        frontier = list(seeds)
        used_edges = set()
        # Alias chunks of the seeds themselves (hop 0 links, weight 1).
        for s in seeds:
            for cid in self.nodes[s]["merge_chunk_ids"]:
                chunk_score[cid] += 1.0
                chunk_path[cid].append({"alias_of": self.nodes[s]["name"], "hop": 0})
        for hop in range(1, self.max_hops + 1):
            nxt = []
            for node in sorted(frontier):
                if node not in seeds and len(self.adj[node]) > self.max_degree:
                    continue
                for edge, other in self.adj[node]:
                    if edge["id"] in used_edges:
                        continue
                    used_edges.add(edge["id"])
                    w = 1.0 / hop
                    chunk_score[edge["chunk_id"]] += w
                    chunk_path[edge["chunk_id"]].append({
                        "hop": hop, "source": self.nodes[edge["source"]]["name"],
                        "relation": edge["relation"], "target": self.nodes[edge["target"]]["name"],
                        "evidence": edge["evidence"]})
                    if other not in visited:
                        visited[other] = hop
                        nxt.append(other)
                        for cid in self.nodes[other]["merge_chunk_ids"]:
                            chunk_score[cid] += w
                            chunk_path[cid].append({"alias_of": self.nodes[other]["name"], "hop": hop})
            frontier = nxt
            if not frontier:
                break
        ranked = sorted(chunk_score.items(), key=lambda kv: (-kv[1], kv[0]))
        results = []
        for cid, score in ranked:
            if cid not in self.chunks:
                continue  # graph built from other chunks than the ones loaded: skip, never invent text
            c = self.chunks[cid]
            results.append({"chunk_id": cid, "text": c["text"], "source": c.get("source", ""),
                            "score": round(score, 4), "rank": len(results) + 1,
                            "retriever": self.name, "path": chunk_path[cid]})
            if len(results) >= top_k:
                break
        return {"entities": [{"id": k, "name": self.nodes[k]["name"], "matched": v}
                             for k, v in sorted(seeds.items())],
                "results": results, "reason": None if results else "no_path"}

    def search(self, question, top_k=10):
        return self.explain(question, top_k)["results"]
