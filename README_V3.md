# RAG V3

## Setup
    pip install -r requirements.txt
    copy .env.example .env      (set GOOGLE_API_KEY)
    python app.py               (http://127.0.0.1:5000)

Add the documents from the Ask page (or the Kestrel files from corpus/), then press "Rebuild index".

## Switching retrievers
UI: the checkboxes dense / BM25 / concept graph on the Ask and Evaluation pages.
Command: python eval_v3.py --corpus kestrel --retrievers dense,bm25,graph
RRF only fuses the retrievers that are switched on. The graph exists for Kestrel only.

## Building the graph
    python graph_cli.py build --corpus kestrel --sleep 1
    python graph_cli.py stats
Rebuild it whenever the chunk size or overlap changes. Optional .env keys: GRAPH_API_KEY, GRAPH_BASE_URL, GRAPH_MODEL, GRAPH_HOPS.

## Grading the triples
    python graph_cli.py sample --n 10 --seed 1
Open eval/results/graph-triples-graded.json, set each grade to correct, wrong or unsupported, then:
    python graph_cli.py grades

## Running the evaluation
    python eval_v3.py --corpus kestrel --corpus own --retrievers dense
    python eval_v3.py --corpus kestrel --corpus own --retrievers dense,bm25
    python eval_v3.py --corpus kestrel --retrievers dense,bm25,graph
    python eval_v3.py --compare FILE_A.json FILE_B.json
Add --answers to also generate the answers (LLM calls). Results go to eval/results/.

## Hidden page
/bm25 is not linked from any page.
