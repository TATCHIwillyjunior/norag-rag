# norag-rag (RAG V4 - Final Release)

This repository contains a local-first, observable, and responsible RAG application
built with dense retrieval, BM25, concept-graph retrieval, reciprocal-rank fusion (RRF),
multi-turn conversation, and Google Gemini.

## V4 Application Features

- **Part A — Observability & Tracing:** Full OpenTelemetry-aligned span hierarchy for every request (`guardrail.pii`, `conversation.rewrite`, `retrieve.dense`, `retrieve.bm25`, `retrieve.graph`, `fuse.rrf`, `generate`, `conversation.suggest`). Open any answer's trace with the **"🔍 Trace Details"** button in the UI, and inspect latency percentiles ($p_{50}$, $p_{95}$), token counts, and costs on the **`/metrics`** dashboard.
- **Part B — Concept Map Explorer (`/explorer`):** Interactive force-directed network visualization of the Kestrel knowledge graph with entity-type filtering, concept search, neighbor traversal, source chunk provenance, and 1-click grounded queries.
- **Part B — Conversation Mode:** Multi-turn conversational memory with LLM query rewriting for follow-up questions (e.g. *"and for the Business plan?"*) and automatic suggestions for the next 2–3 questions.
- **Part C — Responsible AI Guardrails:**
  - **Prompt Injection & Poisoning Scanner (`/quarantine`):** Ingestion guardrail detecting instruction overrides, jailbreaks, hidden HTML instructions, and markdown image exfiltration. Suspicious files are placed into quarantine and isolated from indexing.
  - **PII Redaction Guardrail:** Automatically masks emails, phone numbers, credit cards, and sensitive identifiers before data leaves your machine.

## Web Application Routes

- `/`: Unified Ask & Conversation workspace with Trace Details and follow-up suggestions.
- `/explorer`: Interactive Concept Map Explorer.
- `/metrics`: Live pipeline observability, $p_{50}/p_{95}$ latency breakdown, and cost control.
- `/quarantine`: Responsible AI Quarantine Manager.
- `/eval`: Benchmark evaluation runner across question categories.
- `/words`: Lexical vocabulary inspection.
- `/bm25`: Standalone BM25 challenge page.

## Setup

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

The first start reads every document of the `corpus/` folder (a few minutes, mostly
for the novel); the terminal prints `Ready` when it is finished. Open
<http://127.0.0.1:5000>. Upload more documents on the Ask page if needed. The
evaluation corpora are configured in `eval/corpora.json`:

- `kestrel`: the five Kestrel Markdown documents in `corpus/`
- `own`: `CrimeAndPunishment.txt`

### Try a question in the Ask page

The Ask page accepts a question you type. Select the matching corpus and try
one of these examples:

**Kestrel** (select the Kestrel documents):

- "What does error KST-2044 mean and how do I fix it?" (exact identifier)
- "What happens, step by step, when a customer's card payment fails?"
  (information spread across the billing policy)
- "Can I get a refund on my annual plan 45 days after it renewed?"
  (check the current and previous policy versions)
- "An old alert mentions mailman. Which team looks after that today, and which
  channel should support use?" (multi-hop; enable `dense`, `bm25`, and `graph`)
- "Is there a Kestrel mobile app for iPhone?" (not answered by the corpus)

**Own corpus** (select `CrimeAndPunishment.txt`):

- "What does Raskolnikov take to the pawnbroker?"
- "Who introduces himself as a titular counsellor in the tavern?"
- "Who is Lizaveta in relation to the pawnbroker?"
- "What is the capital of Brazil?" (not answered by this corpus)

These are also examples from the benchmark question files:
[Kestrel](./eval/kestrel.jsonl) and [own](./eval/own.jsonl).

### Run the benchmark questions

The Evaluation page runs the benchmark questions for the selected corpus; you
do not need to enter them individually. The same question sets can be run from
PowerShell with the commands below.

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

### Run the guardrail evaluation

Run the responsible AI guardrail evaluation benchmark:

```powershell
python eval_guardrails.py
```

This runs the 20-sample hostile vs. harmless evaluation, prints the confusion matrix, and outputs metrics to `eval/results/guardrails_eval.json`.

The held-out set (40 samples written before tuning, never tuned on) and the dev set:

```powershell
python eval_guardrails.py --set heldout   # -> eval/results/guardrails_eval_heldout.json
python eval_guardrails.py --set dev
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
