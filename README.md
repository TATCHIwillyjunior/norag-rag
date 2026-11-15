# norag-rag

This repository contains a local RAG application and small standalone demos
for comparing retrieval strategies with Google Gemini. The primary application
is V3: dense retrieval, BM25, concept-graph retrieval, and reciprocal-rank
fusion (RRF).

## V3 application

The Flask app serves the Ask, Evaluation, and Words pages. On the Ask and
Evaluation pages, select the `dense`, `bm25`, and (for Kestrel) `graph`
retrievers. RRF combines only the retrievers selected for a request. The graph
is built for Kestrel only.

### Setup

Requires Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GOOGLE_API_KEY` in `.env` to a key from
[Google AI Studio](https://aistudio.google.com/app/apikey). The key is needed
for Gemini answer generation and graph extraction; retrieval-only evaluation
does not generate answers. Optional graph settings are `GRAPH_API_KEY`,
`GRAPH_BASE_URL`, `GRAPH_MODEL`, and `GRAPH_HOPS`.

### Start the app and build the dense index

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. Documents in `corpus/` are available in the Ask
page; upload documents there if needed, then select **Rebuild index**. The
evaluation corpora are configured in `eval/corpora.json`:

- `kestrel`: the five Kestrel Markdown documents in `corpus/`
- `own`: `CrimeAndPunishment.txt`

After changing corpus files directly or changing the chunk size/overlap,
rebuild the index from the Ask page. Rebuild the Kestrel graph as well after
changing its documents or chunk settings.

### Build and inspect the Kestrel graph

Build the dense index first, then run:

```powershell
python graph_cli.py build --corpus kestrel --sleep 1
python graph_cli.py stats
```

The graph build calls the configured LLM to extract triples. The CLI caches
completed chunk extractions, so a later build can reuse them. The graph
retriever is available for Kestrel only.

To sample triples for manual review, run:

```powershell
python graph_cli.py sample --n 10 --seed 1
```

Edit `eval/results/graph-triples-graded.json`, setting each triple's `grade`
to `correct`, `wrong`, or `unsupported`, then summarize the grades:

```powershell
python graph_cli.py grades
```

### Run the retrieval evaluations

The minimum configurations are:

```powershell
python eval_v3.py --corpus kestrel --corpus own --retrievers dense
python eval_v3.py --corpus kestrel --corpus own --retrievers dense,bm25
python eval_v3.py --corpus kestrel --retrievers dense,bm25,graph
```

The graph configuration applies to Kestrel only. Results are saved under
`eval/results/`. These commands evaluate retrieval without generating answers.
Add `--answers` to make LLM answer-generation calls as well.

Compare two saved result files at Recall@5:

```powershell
python eval_v3.py --compare eval/results/FILE_A.json eval/results/FILE_B.json
```

The tested V3 configuration, graph statistics, benchmark tables, and
limitations are documented in [REPORT.md](./REPORT.md).

## Standalone demos and exercises

These scripts are separate from the V3 Flask app:

- `ask_norag_google.py`: full-context prompting using the Crime and Punishment
  demo text and an injected "needle" passage.
- `ask_rag_google.py`: the original dense ChromaDB RAG demo on
  `docs/CrimeAndPunishment.txt`.
- `bm25/ask_bm25_rag_google.py`: standalone BM25 RAG demo. See
  [bm25/README.md](./bm25/README.md) for usage and its challenge.

The two-hop retrieval exercise is in the
[concept graph challenge](./concepts/concept_graph_challenge_part1.md).
The shortened book text in `docs/CrimeAndPunishment-shorter.txt` is available
for quick experimentation.

## Project structure

- `app.py`: V3 Flask application.
- `corpus/`: documents indexed by the V3 application.
- `eval/`: V3 corpus/question configuration and generated results.
- `graph.py`, `graph_cli.py`: Kestrel graph extraction, retrieval, and CLI.
- `hybrid.py`, `hybrid_eval.py`, `fusion.py`: retriever selection, evaluation,
  and RRF.
- `rag_core.py`: document management, dense ChromaDB indexing, and generation.
- `bm25_index.py`: V3 persistent BM25 index.
- `chroma_db/`: generated persistent dense index.
- `retrieval/`: shared retrieval helpers used by the standalone retrieval
  examples.
- `templates/`: application pages.
- `data/`: generated BM25 cache and graph files.
- `docs/`: source texts used by the standalone demonstrations.
- `tests/`: tests for retrieval and graph functionality.
- `requirements.txt`: Python dependencies.

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```
