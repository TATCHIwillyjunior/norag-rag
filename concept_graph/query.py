"""Ask the graph retriever a question and print the entities, ranked chunks and paths.

  python -m concept_graph.query --graph graph_out/kestrel "Error KST-5031 happened. Which team should support contact?"
"""
import argparse
import json
from pathlib import Path

from .retrieve import GraphRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--graph", required=True, help="folder written by concept_graph.build")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    g = Path(args.graph)
    r = GraphRetriever.load(g / "graph.json", g / "chunks.jsonl", max_hops=args.hops)
    res = r.explain(args.question, args.top_k)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return
    print("Entities:", ", ".join(f"{e['name']} ({e['id']})" for e in res["entities"]) or "none")
    if not res["results"]:
        print("No result:", res["reason"])
    for hit in res["results"]:
        print(f"\n#{hit['rank']}  {hit['chunk_id']}  score={hit['score']}")
        for p in hit["path"]:
            if "alias_of" in p:
                print(f"   hop {p['hop']}: alias link for {p['alias_of']}")
            else:
                print(f"   hop {p['hop']}: {p['source']} -{p['relation']}-> {p['target']}")
        print("   " + hit["text"][:200].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
