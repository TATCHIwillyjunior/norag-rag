# REPORT V3  (write this yourself with your real numbers)

## 0. Team note
I worked alone on this part. (Say who did what, and that the original V2 question set was/was not used.)

## 1. Results
Table: configuration x corpus x question type, HitRate@1/3/5/10 and Recall@1/3/5/10.
Copy the numbers from eval/results/*.json (python eval_v3.py prints them).
Default configuration and why (use the numbers).

## 2. BM25
Where the index lives (data/bm25_index.json), how it is built, tokenization (same as the Words tab), what happens when a document is added or removed (fingerprint changes, index rebuilt).

## 3. Concept graph
Schema (fixed, and why). Entity resolution: one good merge and one wrong or missed (python graph_cli.py stats). Retrieval: BFS, hops, how chunks are ranked, hub filter. Cost: calls, tokens, seconds (graph_cli build output). Failures. Graph in numbers (entities, relations, merges). 10 graded triples: correct / wrong / unsupported counts.

## 4. Fusion
What happened to the graph's chunks after RRF (eval rows: graph_relevant_chunks vs graph_relevant_in_fused_top5).
One question where adding a retriever made it worse (python eval_v3.py --compare A B).
Cost per question per configuration.

## 5. Decision log and known failures
Update the V2 log with the V3 choices.
