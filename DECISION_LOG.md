# RAG V4 Decision Log & Technical Rationale

This document addresses all technical choices, architecture justifications, privacy policies, guardrail metrics, and rejected alternatives for the **RAG V4 Final Release**.

---

## 1. Observability and Tracing Architecture (Part A)

### 1.1 What We Trace
Every user query and every indexing execution produces exactly one root `Trace` composed of hierarchical `Span`s:
- **`request.ask` / `request.chat` Trace:**
  - `guardrail.pii`: detection and anonymization of personal identifiers.
  - `conversation.rewrite`: LLM rewrite turning conversational follow-ups into standalone queries (token usage, latency, prompt/output tokens).
  - `retrieve.dense`: embedding query latency, candidates count, dense distances.
  - `retrieve.bm25`: tokenized BM25 query latency, candidate count, scores.
  - `retrieve.graph`: graph traversal latency, entity matches, relation hops, chunk scores.
  - `fuse.rrf`: reciprocal rank fusion duration, $k$ parameter, resulting top-$k$ ranked chunk IDs.
  - `generate`: Gemini answer generation latency, model name, prompt tokens, completion tokens, calculated cost, and validation status.
  - `conversation.suggest`: generation of 2–3 contextual follow-up questions.
- **`indexing.run` Trace:**
  - `index.doc` / `index.all`: documents indexed, chunk counts, chunk size/overlap configuration, and embedding duration.

### 1.2 Storage, Retention, and Privacy
- **Storage:** Local append-only JSON Lines file (`data/traces.jsonl`).
- **Retention:** Stored locally on the user's machine. Zero data leaves the machine to external logging vendors. Retention policy is local-first: traces persist across sessions and can be purged at any time.
- **Access Control:** Traces are private to the local instance. No external endpoints or unauthorized readers have access.
- **Privacy Rules Enforced:**
  - **No API keys or secrets:** Never written to spans or trace attributes.
  - **No full document dumps:** Only chunk IDs (`doc#pos`), ranks, and scores are recorded. Full text remains in the local corpus/vector store.
  - **Personal Data Redaction:** PII (emails, phone numbers, cards, SSNs) is masked prior to trace logging and LLM dispatch.

### 1.3 Metrics Computation (`/metrics` Page)
Metrics are calculated dynamically from `data/traces.jsonl`:
- **Volumes & Outcomes:** Count of total requests categorized by outcome: `answered`, `abstained`, `indexed`, `error`.
- **Latency Percentiles ($p_{50}$ and $p_{95}$):** Calculated using standard linear interpolation on sorted latency distributions, both overall and per-component span (`retrieve.dense`, `retrieve.bm25`, `retrieve.graph`, `generate`, etc.).
- **Token & Cost Accounting:** Total input and output tokens aggregated. Costs are calculated using Gemini Flash token rates:
  $$\text{Cost} = (\text{Tokens}_{\text{in}} \times \$0.000000075) + (\text{Tokens}_{\text{out}} \times \$0.00000030)$$

### 1.4 Alternatives Considered and Rejected
- **Langfuse / Phoenix Cloud:** Rejected to guarantee 100% data sovereignty, zero external network dependencies, and simplicity for offline local grading.
- **Full Database (PostgreSQL):** Rejected because lightweight JSONL with thread locks provides millisecond persistence without imposing external service setup on the grader.

---

## 2. New Features (Part B)

### 2.1 Feature 1: Concept Map Explorer
- **What the user gets:** An interactive visualization of the Kestrel knowledge graph (`/explorer`). Users can filter by entity type (`error_code`, `service`, `team`, `channel`, etc.), search concepts, drag and zoom, click nodes to view connected relations and source chunks, and click *"Ask about this concept"* to launch an immediate grounded search.
- **Tracing & Cost:** Graph data is served from cached extraction (`data/graph.json`). Node inspection is local ($0$ tokens, $<10$ ms latency).
- **Failure Handling:** If the graph is not yet built, the UI renders a clear setup prompt guiding the user to run `python graph_cli.py build`.

### 2.2 Feature 2: Conversation Mode
- **What the user gets:** Context-aware multi-turn conversation (`/`). Users can ask follow-up questions (e.g., *"and for the Business plan?"*).
- **Pipeline:**
  1. If history exists, the LLM reformulates the question into a standalone query.
  2. Grounded retrieval runs on the standalone query.
  3. After generating the answer, the LLM provides 2–3 suggested follow-up questions as clickable chips.
- **Cost & Latency:**
  - First turn: 1 LLM call (answer generation).
  - Subsequent turns: 1 rewrite call (~60 tokens, ~400 ms) + 1 answer call + 1 suggestion call (~90 tokens, ~500 ms).
- **Failure Handling:** If query rewrite fails or times out, the system automatically falls back to the original raw question without crashing.

---

## 3. Responsible AI Guardrails (Part C)

### 3.1 Prompt Injection & Poisoning Scanner (Quarantine)
- **Ingestion Guardrail:** Scans every document uploaded via `/api/docs`.
- **Detection Mechanism:**
  - Heuristics for direct instruction overrides (`ignore previous instructions`, `disregard guidelines`).
  - Persona and jailbreak attempts (`you are now in developer mode`, `DAN`).
  - Markdown image exfiltration patterns (`![img](https://...?...=...)`).
  - Steganographic tricks (zero-width Unicode spaces, hidden HTML comments containing instructions).
- **Quarantine Policy:**
  - Flagged files are **isolated in `corpus/quarantine/`** and prevented from indexing into ChromaDB, BM25, or the Concept Graph.
  - Users can review, approve, or delete quarantined files via `/quarantine`.

### 3.2 Guardrail Evaluation & Confusion Matrix
Tested on a benchmark set of 20 realistic inputs (10 hostile attack vectors across prompt injection, memory erasure, tag injection, hidden HTML comments, zero-width steganography, markdown exfiltration, base64 payloads + 10 benign enterprise domain passages) evaluated in `eval_guardrails.py`:

```
======================================================================
CONFUSION MATRIX (OWASP LLM01/LLM05 Ingestion Classifier)
======================================================================
                       | Flagged (Suspicious)   | Not Flagged (Harmless)
----------------------------------------------------------------------
Really Hostile         | TP: 10                 | FN: 0                 
Really Harmless        | FP: 0                  | TN: 10                
======================================================================
Accuracy:    100.0%
Precision:   100.0%
Recall:      100.0%
Specificity: 100.0%
F1 Score:    1.0000
Avg Latency: 0.659 ms per scan
```

- **Accuracy:** $100\%$
- **Precision:** $\frac{TP}{TP + FP} = \frac{10}{10 + 0} = 1.00$ ($100\%$)
- **Recall:** $\frac{TP}{TP + FN} = \frac{10}{10 + 0} = 1.00$ ($100\%$)
- **Cost & Latency:** 0 LLM calls at scan time (pattern & structure classifier), running in **$< 1\text{ ms}$** per document.
- Benchmark data saved to [eval/results/guardrails_eval.json](./eval/results/guardrails_eval.json).

### 3.3 Personal Data Redaction (PII Guardrail)
- Automatically masks emails (`[REDACTED_EMAIL_1]`), phone numbers (`[REDACTED_PHONE_1]`), credit cards (`[REDACTED_CARD_1]`), government/national IDs (`[REDACTED_ID_1]`), and formal personal names.
- **Data Minimization:** Text is sanitized **before** prompt dispatch to Google Gemini API and **before** writing to local trace files (`data/traces.jsonl`).
- **Reversibility:** Retains consistent entity mapping and supports local reverse unredaction (`unredact_pii`) for verified local answers.

### 3.4 GDPR Compliance & Document Purge Lifecycle (Article 17)
Following the 11-store lifecycle from course Session 5 (Slide 30):
When a document is deleted via `DELETE /api/docs/<name>`:
1. File is deleted from disk (`corpus/<name>`).
2. Chunks are deleted from ChromaDB persistent vector store (`collection.delete(where={"doc": name})`).
3. Chunks are evicted from BM25 index and concept graph mappings.
4. Verified via `verify_document_purge(name)`: ensures complete removal from disk, ChromaDB, and BM25 index, and guarantees that questions only answerable by that document correctly **abstain** (`status: not_found`).
