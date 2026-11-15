import argparse
import json

import evalkit
import hybrid
import hybrid_eval
import v2
from rag_core import RagError


def table(summary):
    header = ["type", "n"] + [f"hit@{k}" for k in evalkit.KS] + [f"recall@{k}" for k in evalkit.KS] + ["mrr"]
    print("\t".join(header))
    lines = [("OVERALL", summary["overall"])] + list(summary["per_type"].items())
    for name, a in lines:
        if not a:
            continue
        values = [name, str(a["n"])]
        values += [f"{a[f'hit@{k}']:.2f}" for k in evalkit.KS]
        values += [f"{a[f'recall@{k}']:.2f}" for k in evalkit.KS]
        values.append(f"{a['mrr']:.2f}")
        print("\t".join(values))


def print_result(r):
    print(f"\n== {r['corpus']} | retrievers: {'+'.join(r['retrievers'])} | chunk size {r['chunk_size']}, overlap {r['overlap_percent']}% | candidates {r['candidates']}, rrf k {r['rrf_k']}")
    for p in r["evidence_problems"]:
        print("WARNING evidence:", p)
    print("\nAFTER FUSION")
    table(r["summary"])
    if len(r["retrievers"]) > 1:
        for name, s in r["per_retriever_summary"].items():
            print(f"\nretriever alone: {name}")
            table(s)
    print("\ncost:", json.dumps(r["cost"]))
    print("\nper question (evidence ranks after fusion):")
    for row in r["rows"]:
        if row["answerable"]:
            extra = ""
            if "graph_relevant_chunks" in row:
                extra = f"  graph found {len(row['graph_relevant_chunks'])} relevant, {len(row['graph_relevant_in_fused_top5'])} reached top 5"
            print(f"{row['id']}\t{row['type']}\t{row['evidence_ranks']}{extra}")
        else:
            print(f"{row['id']}\t{row['type']}\tunanswerable")
    print("\nsaved to", r["file"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", action="append", help="corpus name from eval/corpora.json (repeatable, default all)")
    parser.add_argument("--retrievers", default="dense,bm25", help="comma list among dense,bm25,graph")
    parser.add_argument("--candidates", type=int, default=hybrid.CANDIDATES)
    parser.add_argument("--rrf-k", type=int, default=hybrid.RRF_K)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--answers", action="store_true", help="also generate the answers (calls the LLM)")
    parser.add_argument("--answer-k", type=int, default=5)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two result files: questions that got worse or better")
    parser.add_argument("--compare-k", type=int, default=5)
    args = parser.parse_args()

    if args.compare:
        out = hybrid_eval.compare(args.compare[0], args.compare[1], args.compare_k)
        print(f"A = {out['A']}   B = {out['B']}   (recall@{out['k']})")
        print("\nWORSE in B:")
        for item in out["worse_in_B"]:
            print(" ", json.dumps(item, ensure_ascii=False))
        print("\nBETTER in B:")
        for item in out["better_in_B"]:
            print(" ", json.dumps(item, ensure_ascii=False))
        return

    if (args.chunk_size is None) != (args.overlap is None):
        parser.error("--chunk-size and --overlap go together")
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    corpora = args.corpus or list(v2.load_manifest().keys())
    try:
        if args.chunk_size is not None:
            print("Rebuilding the index... (the graph must be rebuilt too if you want to use it)")
            v2.reconfigure(args.chunk_size, args.overlap)
        for name in corpora:
            if "graph" in retrievers and name != "kestrel":
                print(f"Skipping {name}: the graph only exists for the Kestrel corpus. Use --retrievers dense,bm25 for it.")
                continue
            print_result(hybrid_eval.run_eval(name, retrievers, args.answers, args.answer_k, args.candidates, args.rrf_k))
    except RagError as e:
        print("Error:", e.message)


if __name__ == "__main__":
    main()
