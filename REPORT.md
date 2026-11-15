# RAG V3 Evaluation Report

Run date: 2026-11-08

## 0. Team note and evaluation scope

This report describes the V3 implementation and local evaluation run. The
repository does not record contributor-by-contributor ownership. The original
V2 question set was not used; the runs used the V3 corpora and questions
configured in `eval/corpora.json` (`eval/kestrel.jsonl` and `eval/own.jsonl`).

All retrieval runs used 800-character chunks, 10% overlap, candidate depth 20,
and RRF constant 60. The Kestrel set has 13 questions, of which 11 are
answerable from the corpus; the own set has 10 questions, of which 8 are
answerable. The reported retrieval metrics use answerable questions only.
Answer generation was not enabled, so these results measure retrieval and
evidence recall, not answer quality.

## 1. Results

Values are copied from the saved `eval/results/*.json` files and rounded to
four decimal places. `Hit@k` is whether at least one evidence chunk was found
in the top *k*; `Recall@k` is the fraction of the question's evidence chunks
found in the top *k*. MRR is mean reciprocal rank of the first relevant
chunk. `n` is the number of answerable questions in each row.

### Kestrel

| Configuration | Question type | n | Hit@1 / 3 / 5 / 10 | Recall@1 / 3 / 5 / 10 | MRR |
|---|---|---:|---|---|---:|
| dense | overall | 11 | 0.6364 / 0.8182 / 0.9091 / 1.0000 | 0.4015 / 0.6970 / 0.7424 / 0.9091 | 0.7478 |
| dense | exact_identifier | 2 | 0.5000 / 0.5000 / 0.5000 / 1.0000 | 0.2500 / 0.5000 / 0.5000 / 1.0000 | 0.5714 |
| dense | split_answer | 2 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 0.6250 / 1.0000 / 1.0000 / 1.0000 | 1.0000 |
| dense | two_hop | 4 | 0.7500 / 0.7500 / 1.0000 / 1.0000 | 0.4167 / 0.4167 / 0.5417 / 0.7500 | 0.8125 |
| dense | versions | 2 | 0.0000 / 1.0000 / 1.0000 / 1.0000 | 0.0000 / 1.0000 / 1.0000 / 1.0000 | 0.4167 |
| dense | vocabulary | 1 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 1.0000 |
| dense+bm25 | overall | 11 | 0.5455 / 0.8182 / 0.9091 / 1.0000 | 0.2652 / 0.5833 / 0.7197 / 0.9091 | 0.6705 |
| dense+bm25 | exact_identifier | 2 | 0.5000 / 0.5000 / 1.0000 / 1.0000 | 0.2500 / 0.5000 / 1.0000 / 1.0000 | 0.6250 |
| dense+bm25 | split_answer | 2 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 0.3750 / 0.6250 / 0.6250 / 1.0000 | 1.0000 |
| dense+bm25 | two_hop | 4 | 0.7500 / 1.0000 / 1.0000 / 1.0000 | 0.4167 / 0.5417 / 0.6667 / 0.7500 | 0.8333 |
| dense+bm25 | versions | 2 | 0.0000 / 1.0000 / 1.0000 / 1.0000 | 0.0000 / 1.0000 / 1.0000 / 1.0000 | 0.3333 |
| dense+bm25 | vocabulary | 1 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.1250 |
| dense+bm25+graph | overall | 11 | 0.7273 / 0.9091 / 0.9091 / 1.0000 | 0.4015 / 0.7652 / 0.7955 / 1.0000 | 0.7992 |
| dense+bm25+graph | exact_identifier | 2 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 0.7500 / 1.0000 / 1.0000 / 1.0000 | 1.0000 |
| dense+bm25+graph | split_answer | 2 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 0.3750 / 0.6250 / 0.6250 / 1.0000 | 1.0000 |
| dense+bm25+graph | two_hop | 4 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 0.5417 / 0.7917 / 0.8750 / 1.0000 | 1.0000 |
| dense+bm25+graph | versions | 2 | 0.0000 / 1.0000 / 1.0000 / 1.0000 | 0.0000 / 1.0000 / 1.0000 / 1.0000 | 0.3333 |
| dense+bm25+graph | vocabulary | 1 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.0000 / 0.0000 / 0.0000 / 1.0000 | 0.1250 |

### Own corpus (Crime and Punishment)

| Configuration | Question type | n | Hit@1 / 3 / 5 / 10 | Recall@1 / 3 / 5 / 10 | MRR |
|---|---|---:|---|---|---:|
| dense | overall | 8 | 0.1250 / 0.3750 / 0.3750 / 0.3750 | 0.1250 / 0.3125 / 0.3125 / 0.3125 | 0.2500 |
| dense | exact_identifier | 3 | 0.3333 / 0.3333 / 0.3333 / 0.3333 | 0.3333 / 0.3333 / 0.3333 / 0.3333 | 0.3333 |
| dense | other | 2 | 0.0000 / 0.5000 / 0.5000 / 0.5000 | 0.0000 / 0.5000 / 0.5000 / 0.5000 | 0.2500 |
| dense | split_answer | 1 | 0.0000 / 1.0000 / 1.0000 / 1.0000 | 0.0000 / 0.5000 / 0.5000 / 0.5000 | 0.5000 |
| dense | vocabulary | 2 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 |
| dense+bm25 | overall | 8 | 0.3750 / 0.5000 / 0.5000 / 0.6250 | 0.3125 / 0.4375 / 0.5000 / 0.6250 | 0.4345 |
| dense+bm25 | exact_identifier | 3 | 0.3333 / 0.3333 / 0.3333 / 0.6667 | 0.3333 / 0.3333 / 0.3333 / 0.6667 | 0.3810 |
| dense+bm25 | other | 2 | 0.5000 / 1.0000 / 1.0000 / 1.0000 | 0.5000 / 1.0000 / 1.0000 / 1.0000 | 0.6667 |
| dense+bm25 | split_answer | 1 | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 0.5000 / 0.5000 / 1.0000 / 1.0000 | 1.0000 |
| dense+bm25 | vocabulary | 2 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0.0000 |

### Configuration choice

For Kestrel, dense+BM25+graph had the best measured overall Recall@5 (0.7955)
and MRR (0.7992), versus 0.7424 / 0.7478 for dense and 0.7197 / 0.6705 for
dense+BM25. Hit@5 was 0.9091 for all three configurations. For the own corpus,
dense+BM25 outperformed dense on Recall@5 (0.5000 versus 0.3125) and MRR
(0.4345 versus 0.2500). The graph is Kestrel-only, so dense+BM25 is the
common configuration when one configuration must support both corpora. These
are small test sets; treat the differences as observed results, not proof of
general superiority.

## 2. BM25

The persistent BM25 cache is `data/bm25_index.json`. It is built from the
current shared chunks and stores each chunk's document, token length, and term
frequencies. Tokenization is the same `\w+` tokenizer used by the Words
functionality, lowercased before splitting. BM25 uses `k1=1.5` and `b=0.75`.
Search filters the cached index to the selected documents and returns the
highest-scoring chunks.

The cache fingerprint hashes the sorted chunk IDs and chunk text. Adding or
removing a document, changing document text, or changing chunk settings changes
the fingerprint; the next search rebuilds the cached index. In the current
workspace the index covers 1,647 chunks across the six indexed documents.

## 3. Concept graph

### Schema and graph size

The graph uses a fixed typed schema. Entity types are `error_code`, `service`,
`team`, `channel`, `plan`, `feature`, `policy`, `role`, `setting`, and `other`.
Relations are `owned_by`, `escalates_to`, `raised_by`, `on_call_channel`,
`available_on`, `requires`, `replaces`, `configured_in`, `applies_to`, and
`related_to`. Keeping the schema constrained makes extraction output
validatable and keeps relation labels consistent across the Kestrel documents.

The graph was built from five Kestrel documents, with 800-character chunks and
10% overlap, using `gemini-3.5-flash-lite`.

| Graph statistic | Value |
|---|---:|
| Chunks processed | 43 |
| Entities | 47 |
| Relations (edges) | 38 |
| Extracted triples | 43 |
| Candidate entity merges | 0 |
| Invalid triples | 0 |
| Failed chunks | 0 |

Entity types: 8 services, 7 teams, 7 channels, 7 error codes, 5 plans, 4
settings, 4 features, 2 policies, 2 roles, and 1 other. Relation counts:
7 `owned_by`, 6 `on_call_channel`, 6 `requires`, 5 `available_on`,
4 `applies_to`, 4 `related_to`, 3 `escalates_to`, 2 `replaces`, and 1
`raised_by`; no `configured_in` edge was extracted.

There were no candidate merges, so this run provides no example of a successful
or incorrect entity merge. No merge example is inferred or invented.

### Retrieval behavior

Graph retrieval finds graph entity names/forms in the question, ignores generic
entities appearing in more than 25% of graph chunks, and traverses the graph
with breadth-first search up to the default three hops. Traversal can follow
edges from either endpoint. A chunk receives a score of `1 / hop distance` for
each reached edge that cites it; scores are accumulated and chunks are ranked
by score, with chunk ID as a deterministic tie-breaker. Stale graph chunk
hashes are excluded.

### Build cost and triple grading

The complete graph build used 43 LLM calls, 23,003 input tokens, 2,009 output
tokens, and 188.3 seconds of recorded extraction time, with zero retries. The
successful resume run made 16 calls and reused 27 cached extractions. An
initial attempt encountered an API 503; the cached extractions allowed the
build to resume, and the final graph reports zero failed chunks.

The 10 sampled triples were graded as follows:

| Correct | Wrong | Unsupported | Not graded | Total |
|---:|---:|---:|---:|---:|
| 9 | 0 | 1 | 0 | 10 |

## 4. Fusion

The evaluation uses reciprocal-rank fusion (RRF) with candidate depth 20 and
constant 60. On Kestrel, adding the graph to dense+BM25 raised Recall@5 from
0.7197 to 0.7955 and MRR from 0.6705 to 0.7992; Hit@5 remained 0.9091. The
graph-only retriever had Hit@5 0.4545 and Recall@5 0.3788, so the result is
for the fused configuration, not the graph alone.

Across answerable graph-evaluation questions, graph retrieval returned 11
relevant chunks for six questions. Ten of those 11 chunks appeared in the
fused top five:

| Question | Relevant graph chunks | In fused top 5 |
|---|---:|---:|
| q01 | 1 | 1 |
| q04 | 2 | 2 |
| q08 | 2 | 1 |
| q11 | 2 | 2 |
| q12 | 2 | 2 |
| q13 | 2 | 2 |

The graph comparison at Recall@5 found no answerable question that worsened
when graph was added to dense+BM25. It improved q12 from 0.6667 to 1.0000 and
q13 from 0.5000 to 1.0000. Adding BM25 to dense was mixed on Kestrel: q03
("What happens, step by step, when a customer's card payment fails?") fell
from Recall@5 1.0000 to 0.2500, and q05 ("I lost my phone and I can't log in
anymore. What can I do?") fell from 1.0000 to 0.0000. In contrast, q01
improved from 0.0000 to 1.0000 and q11 from 0.5000 to 1.0000.

Average retrieval latency from each saved run was:

| Corpus | Configuration | Average retrieval time | Retrieval LLM calls per question |
|---|---|---:|---:|
| Kestrel | dense | 637 ms | 0 |
| Kestrel | dense+bm25 | 1,205 ms | 0 |
| Kestrel | dense+bm25+graph | 710 ms | 0 |
| own | dense | 678 ms | 0 |
| own | dense+bm25 | 825 ms | 0 |

These are observed run averages, not a controlled latency benchmark; cache and
machine conditions can affect them. Graph extraction calls are reported
separately above.

## 5. Decision log and known limitations

- Keep chunk size at 800 and overlap at 10% so dense, BM25, graph, and evaluation
  use the same chunk identities.
- Use dense+BM25+graph for Kestrel when optimizing for the measured Recall@5;
  use dense+BM25 for the own corpus. Use dense+BM25 as the shared
  graph-independent setup.
- Keep the graph limited to Kestrel: that is the corpus used to build and
  evaluate it.
- The test sets are small (11 answerable Kestrel questions and 8 answerable
  own-corpus questions), and graph grading covers only 10 sampled triples.
  Results do not establish statistical significance or answer-generation
  quality.
- The graph produced no entity merge candidates, so entity-resolution quality
  could not be assessed from this build.
- The initial graph build hit an API 503, but the cached build resumed and
  completed with no failed chunks.

## 6. RAG V4 Final Release: Observability, Features, and Responsible AI

The V4 release expands V3 into a production-grade observable and responsible system:

### 6.1 Observability & Tracing (Part A)
- **Hierarchical Spans:** Every query generates a trace with discrete spans (`guardrail.pii`, `conversation.rewrite`, `retrieve.dense`, `retrieve.bm25`, `retrieve.graph`, `fuse.rrf`, `generate`, `conversation.suggest`).
- **UI Trace Viewer:** Every answer includes a "🔍 Trace Details" drawer showing the execution breakdown, latency, tokens, cost, and retrieved chunk IDs.
- **Metrics Dashboard (`/metrics`):** Displays real-time request counts, outcome distribution (answered, abstained, error), $p_{50}$ and $p_{95}$ latencies overall and by span, and token/cost aggregates.

### 6.2 New Features (Part B)
- **Concept Map Explorer (`/explorer`):** Interactive force-directed network view of the 47 nodes and 38 edges in the Kestrel concept graph. Features entity type filtering, concept search, neighbor traversal, source chunk provenance, and 1-click query launching.
- **Conversation Mode (`/`):** Multi-turn conversational memory with LLM query rewriting for follow-up questions and automatic generation of 2–3 suggested next questions.

### 6.3 Responsible AI Guardrails (Part C)
- **Injection Scanner & Quarantine (`/quarantine`):** Ingestion guardrail detecting direct instruction overrides, jailbreaks, hidden HTML instructions, and markdown image exfiltration. Flags place documents into an isolated quarantine folder (`corpus/quarantine/`).
- **PII Redaction Guardrail:** Automatically masks emails, phone numbers, credit cards, and sensitive identifiers before data leaves the local environment.

### 6.4 Cost & Latency Benchmark (V4 Pipeline)

| Pipeline Step / Feature | Typ. Latency (p50) | In Tokens | Out Tokens | Cost per Action | Failure Handling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **First Turn (Standard Ask)** | ~1,350 ms | ~1,800 | ~90 | ~$0.00016 | Abstains if no chunks; validates schema |
| **Follow-up Turn (Conversation)** | ~2,100 ms | ~2,200 | ~150 | ~$0.00021 | Falls back to raw query if rewrite fails |
| **Concept Map Inspection** | < 5 ms | 0 | 0 | $0.00000 | Serves from local index; node error fallback |
| **Injection Scan (Ingestion)** | < 1 ms | 0 | 0 | $0.00000 | Quarantines file; notifies user |
| **PII Redaction Guardrail** | < 1 ms | 0 | 0 | $0.00000 | Masks identifiers; records span |

### 6.5 Guardrail Performance & Confusion Matrix

Evaluated on 20 realistic inputs (10 hostile attack payloads + 10 benign enterprise documents) in `eval_guardrails.py`:

```
======================================================================
CONFUSION MATRIX (OWASP LLM01/LLM05 Ingestion Classifier)
======================================================================
                       | Flagged (Suspicious)   | Not Flagged (Harmless)
----------------------------------------------------------------------
Really Hostile         | TP: 10                 | FN: 0                 
Really Harmless        | FP: 0                  | TN: 10                
======================================================================
Accuracy: 100.0% | Precision: 100.0% | Recall: 100.0% | F1 Score: 1.0000
Average Latency: 0.659 ms per document scan
```

### 6.6 18-Minute Presentation Runbook (Team of 3)

The team defense follows the course schedule (Slide 52: 4 min demo + 3x3 min individual talks + 5 min Q&A):

1. **Live Demo (4 minutes):**
   - *Query with Traces (1.5 min):* Ask *"What happens, step by step, when a customer payment fails?"*, show the grounded answer, citations, and click **`Details ▸ trace 7f3a`** to walk through the span tree (Dense + BM25 + RRF + Generate), latency, and costs.
   - *Conversation Mode (1 min):* Ask a follow-up *"and what if the card is expired?"*, show the automatic query rewrite badge and the suggested next questions chips.
   - *Concept Explorer (1 min):* Switch to `/explorer`, filter by `error_code`, click `KST-2044`, show its connected relations and source chunk quotes, and launch a query from the node.
   - *Quarantine Ingestion (0.5 min):* Upload a test file with prompt injection, show the immediate quarantine warning, and open `/quarantine` to inspect and manage it.
2. **Individual Presentations (3 minutes each):**
   - **Willy (Observability & Architecture):** Design of the OpenTelemetry-aligned span hierarchy, local JSONL persistence, cost tracking formula, and the `/metrics` dashboard ($p_{50}$ / $p_{95}$ percentiles).
   - **Fred (New Features & User Experience):** UX clean-up from V1–V3 clutter, Concept Map Explorer force-directed graph architecture, and Conversation Mode query rewriting.
   - **Keylan (Responsible AI & Compliance):** Prompt injection / poisoning heuristics, the quarantine isolation workflow, PII redaction engine, and GDPR Article 17 document deletion lifecycle.
3. **Q&A Preparation (5 minutes):**
   - *Why local JSONL over Langfuse/Phoenix?* Data sovereignty, zero external latency overhead, and 100% offline grading reproducibility.
   - *What happens when query rewriting fails?* Graceful fallback to the raw query; user never experiences a crash.
   - *How do you prevent data exfiltration?* Ingestion scanner blocks markdown image links and scripts, and PII redactor scrubs emails/numbers before Gemini API transmission.

Detailed architecture rationales and guardrail confusion matrices are documented in [DECISION_LOG.md](./DECISION_LOG.md).

