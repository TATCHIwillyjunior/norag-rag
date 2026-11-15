import argparse
import json

import graph
import v2
from rag_core import RagError


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="extract triples with the LLM and build the graph")
    b.add_argument("--corpus", default="kestrel")
    b.add_argument("--sleep", type=float, default=1.0, help="seconds to wait between LLM calls")
    b.add_argument("--limit", type=int, help="only the first N chunks (to test cheaply)")
    sub.add_parser("stats", help="entities, relations, merges, build cost")
    s = sub.add_parser("sample", help="pick random triples to grade by hand")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--seed", type=int, default=1)
    sub.add_parser("grades", help="count correct / wrong / unsupported in the graded file")
    args = parser.parse_args()

    try:
        if args.cmd == "build":
            spec = v2.corpus_spec(args.corpus)
            chunks = v2.all_chunks(args.corpus)
            if not chunks:
                print("No chunks. Add the documents and rebuild the index first.")
                return
            if args.limit:
                chunks = chunks[: args.limit]
            g = graph.build_graph(chunks, spec["docs"], sleep=args.sleep)
            print(json.dumps(g["stats"], indent=1))
            print("total build cost:", json.dumps(g["build_cost_total"]))
            print("this run:", json.dumps(g["last_run"]))
        elif args.cmd == "stats":
            st = graph.stats()
            if st is None:
                print("No graph yet. Run: python graph_cli.py build")
                return
            print(json.dumps({k: v for k, v in st.items() if k not in ("merges", "invalid_triples")}, indent=1))
            print(f"\n{len(st['merges'])} entities have several written forms or types (candidate merges):")
            for m in st["merges"][:40]:
                print(" ", json.dumps(m, ensure_ascii=False))
            print(f"\n{len(st['invalid_triples'])} triples were rejected by the schema:")
            for t in st["invalid_triples"][:20]:
                print(" ", json.dumps(t, ensure_ascii=False))
        elif args.cmd == "sample":
            by_id = {c["id"]: c for c in v2.all_chunks()}
            data = graph.sample_triples(args.n, args.seed, by_id)
            graph.GRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
            if graph.GRADES_FILE.exists():
                print(f"{graph.GRADES_FILE} already exists and was NOT overwritten (it may hold your grades). Delete it first to resample.")
                return
            graph.GRADES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved {data['n']} triples to {graph.GRADES_FILE}. Open it and set grade to correct, wrong or unsupported.")
        elif args.cmd == "grades":
            print(json.dumps(graph.grade_counts()))
    except RagError as e:
        print("Error:", e.message)


if __name__ == "__main__":
    main()
