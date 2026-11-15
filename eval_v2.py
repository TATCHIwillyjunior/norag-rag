import argparse

import evalkit
import v2
from rag_core import RagError


def print_summary(result):
    s = result["summary"]
    print(f"\n== {result['corpus']}  chunk size {result['chunk_size']}, overlap {result['overlap_percent']}%")
    print(f"answerable: {s['answerable_questions']}, unanswerable (left out): {s['unanswerable_questions']}")
    for p in result["evidence_problems"]:
        print("WARNING evidence:", p)
    header = ["type", "n"] + [f"hit@{k}" for k in evalkit.KS] + [f"recall@{k}" for k in evalkit.KS] + ["mrr"]
    print("\t".join(header))
    lines = [("OVERALL", s["overall"])] + list(s["per_type"].items())
    for name, a in lines:
        if not a:
            continue
        values = [name, str(a["n"])]
        values += [f"{a[f'hit@{k}']:.2f}" for k in evalkit.KS]
        values += [f"{a[f'recall@{k}']:.2f}" for k in evalkit.KS]
        values.append(f"{a['mrr']:.2f}")
        print("\t".join(values))
    print("\nper question (evidence ranks, None = not in the top 10):")
    for r in result["rows"]:
        if r["answerable"]:
            print(f"{r['id']}\t{r['type']}\t{r['evidence_ranks']}")
        else:
            print(f"{r['id']}\t{r['type']}\tunanswerable")
    print("\nsaved to", result["file"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", action="append", help="corpus name from eval/corpora.json (repeatable, default all)")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    parser.add_argument("--answers", action="store_true", help="also generate the answers (calls the LLM)")
    parser.add_argument("--answer-k", type=int, default=5)
    parser.add_argument("--no-context", action="store_true", help="ask the questions to the LLM without retrieval")
    args = parser.parse_args()

    if (args.chunk_size is None) != (args.overlap is None):
        parser.error("--chunk-size and --overlap go together")
    corpora = args.corpus or list(v2.load_manifest().keys())

    try:
        if args.chunk_size is not None:
            print("Rebuilding the index...")
            v2.reconfigure(args.chunk_size, args.overlap)
        for name in corpora:
            if args.no_context:
                result = v2.no_context_test(name)
                print(f"no-context answers for {name} saved to {result['file']}")
                print("Open the file and set model_knew_it to true or false for each question.")
            else:
                print_summary(v2.run_eval(name, with_answers=args.answers, answer_k=args.answer_k))
    except RagError as e:
        print("Error:", e.message)


if __name__ == "__main__":
    main()
