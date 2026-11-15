"""Graph quality check: 10 random triples, graded by hand against their source chunk.

  python -m concept_graph.triples sample  --graph graph_out/kestrel [--n 10 --seed 42]
      -> graph_out/kestrel/triple_grades.jsonl (one row per triple, "grade" empty)
  then open the file, read each chunk_text, and set "grade" to correct / wrong / unsupported
  (and "note" to say why), then:
  python -m concept_graph.triples summary --graph graph_out/kestrel
      -> prints the totals and writes triple_grades_summary.json

Grades:
  correct      the chunk states this fact, with these entities and this relation direction
  wrong        the chunk talks about it but the triple gets it wrong (wrong entity, relation or direction)
  unsupported  the chunk does not state this fact (even if it is true elsewhere)
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

GRADES = {"correct", "wrong", "unsupported"}


def sample(graph_dir, n=10, seed=42):
    g = Path(graph_dir)
    graph = json.loads((g / "graph.json").read_text(encoding="utf-8"))
    nodes = {x["id"]: x for x in graph["nodes"]}
    chunks = {}
    for line in (g / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    picked = random.Random(seed).sample(graph["edges"], min(n, len(graph["edges"])))
    rows = []
    for e in picked:
        rows.append({
            "edge_id": e["id"],
            "triple": f"({nodes[e['source']]['name']}, {e['relation']}, {nodes[e['target']]['name']})",
            "subject_type": nodes[e["source"]]["type"], "object_type": nodes[e["target"]]["type"],
            "chunk_id": e["chunk_id"], "evidence": e["evidence"],
            "chunk_text": chunks.get(e["chunk_id"], {}).get("text", ""),
            "grade": "", "note": ""})
    out = g / "triple_grades.jsonl"
    if out.exists():
        raise SystemExit(f"{out} already exists: delete it first if you really want a new sample")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"seed": seed, "n": len(rows), "population": len(graph["edges"])}
    (g / "triple_grades_sample.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return out, rows


def summary(graph_dir):
    g = Path(graph_dir)
    rows = [json.loads(l) for l in (g / "triple_grades.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    bad = [r["edge_id"] for r in rows if r["grade"] not in GRADES]
    if bad:
        raise SystemExit(f"not graded yet (or invalid grade): {bad}")
    totals = Counter(r["grade"] for r in rows)
    res = {"n": len(rows), **{k: totals.get(k, 0) for k in sorted(GRADES)},
           "precision": round(totals.get("correct", 0) / len(rows), 3)}
    (g / "triple_grades_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "summary"])
    ap.add_argument("--graph", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.cmd == "sample":
        out, rows = sample(args.graph, args.n, args.seed)
        for r in rows:
            print(r["edge_id"], r["triple"], "<-", r["chunk_id"])
        print(f"\nNow grade them in {out}")
    else:
        print(json.dumps(summary(args.graph), indent=1))


if __name__ == "__main__":
    main()
