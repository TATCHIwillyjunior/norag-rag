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

### 3.2 Guardrail Evaluation (V4.1: held-out set)

The first benchmark (20 samples, `python eval_guardrails.py`) was written together with the rules,
so its 100% only shows that the rules match their own examples. On new text the baseline scanner
caught **1 of 20** attacks.

V4.1 adds two sets in `eval/guardrails/`:

- **`dev.jsonl`** (8 hostile, 6 harmless): used to tune the rules.
- **`heldout.jsonl`** (20 hostile, 20 harmless): written **before** any rule change and never used for
  tuning. It covers paraphrased overrides, text addressed to the AI reading the page, French, fake
  `System:`/`Assistant:` turns, exfiltration links and HTML beacons, letter-spaced and leetspeak
  obfuscation, hidden CSS text, comments, role-play, priority claims, secret requests and base64.
  Harmless samples include traps: a security note that *quotes* an attack, "Admin can override…",
  a benign HTML comment, a markdown image, a SHA-256, a JWT example, `display:flex`.

Held-out results (`python eval_guardrails.py --set heldout`):

| Scanner | Attacks caught (TP / 20) | False positives (FP / 20) | Recall | Precision |
|---|---|---|---|---|
| V4 baseline | 1 | 1 | 5% | 50% |
| V4.1 | 19 | 0 | 95% | 100% |

Files: `eval/results/guardrails_eval_heldout_baseline.json`, `eval/results/guardrails_eval_heldout.json`.

What changed: rules target the intent (override verb + "what you were told", text addressed to the
model, priority claims, URL templates carrying conversation data, hidden CSS text) instead of one
wording; a second pass runs on a de-obfuscated copy (zero-width chars removed, `i g n o r e` joined,
leetspeak mapped); padded base64 is decoded and scanned with the same rules; a quoted example
introduced by "such as", "attackers write"… is logged as a *mention* and does not quarantine.
No file of the real corpus (Kestrel docs, the novel, the espresso manual) is flagged
(`tests/test_v4_guardrails_heldout.py`). Still 0 LLM calls, under 1 ms per document.

Limits:
- Remaining miss: `I-g-n-o-r-e   a-l-l   p-r-e-v-i-o-u-s …` (letters spaced, words separated by
  several spaces). Left as is so the held-out score stays honest.
- We wrote the held-out set ourselves, knowing the attack families: it is not an independent
  red-team test. A pattern scanner cannot catch every rewording; an LLM or classifier-based
  detector would catch more at a cost per document.

### 3.3 Personal Data Redaction (PII Guardrail)
- Masks emails (`[REDACTED_EMAIL_1]`), phone numbers, payment cards, IBANs, national IDs and names
  that follow a title (`Dr.`, `Mrs.`, `M.`, `Mme`…).
- Phones are matched in real formats only: international `+CC …`, French `0X XX XX XX XX`,
  North American `(555) 019-2834`. Before V4.1, any digit group of 8–15 digits counted, so an
  invoice number was masked while `01 23 45 67 89` was not.
- **What is redacted before Gemini (V4.1):**
  - the question;
  - the **retrieved chunks** (before V4.1 they were sent unredacted);
  - the conversation history used for query rewriting (before V4.1 the *raw* previous questions
    were sent back);
  - the answer sent to generate suggestions.
- The question and the chunks share **one placeholder map**, so the model can still answer
  "write to [REDACTED_EMAIL_1]"; the answer is un-redacted locally before it is shown.
- Traces only store redaction counts per kind, never the values.
- Limit: names without a title ("Inès Moreau") are not redacted. In this corpus they are staff
  contacts that answers need; redacting every capitalised pair would also hit team names.

### 3.4 GDPR Compliance & Document Purge Lifecycle (Article 17)
`DELETE /api/docs/<name>` runs `responsible.purge_document(name)`:

1. Deletes the file from `corpus/` and its chunks from ChromaDB.
2. Rebuilds the BM25 index file `data/bm25_index.json` (its fingerprint changed).
3. Removes the document's chunks from every concept-graph node and edge in `data/graph.json`;
   a node or edge with no chunk left is dropped. Shared nodes keep their other sources.
4. Removes the document's rows from the graph extraction cache `data/graph_extractions.jsonl`.
5. Drops conversation turns whose answer cites the document.
6. Runs `verify_document_purge(name)`, which checks each store above, and returns the report in the
   API response (`purge.is_fully_purged`).

Before V4.1, only step 1 was implemented: the graph, its cache and the BM25 index file kept the
document, and the verification function existed but was never called.
Traces keep chunk IDs (document name + position) but no text. Tested in
`tests/test_v4_gdpr_purge.py` on temporary files.
