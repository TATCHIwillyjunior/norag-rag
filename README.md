# norag-rag project

This workspace contains Python applications to compare **NoRAG** (full-context prompting) versus **RAG** (retrieval-augmented generation) using **Google Gemini** via the **OpenAI SDK**.

For the BM25 lexical-retrieval example, see [bm25/README.md](./bm25/README.md).
For the two-hop retrieval exercise, see the [concept graph challenge](./concepts/concept_graph_challenge_part1.md).

## Applications

1. `ask_norag_google.py`
- NoRAG with Google Gemini via OpenAI SDK (`gemini-3.5-flash-lite`).
- Loads `docs/CrimeAndPunishment.txt`, injects a "needle" passage, and runs two direct large-context questions:
	- a baseline question about Marmeladov (present in the original book)
	- a needle question about Denis (the injected passage)

2. `ask_rag_google.py`
- RAG with Google Gemini for generation via OpenAI SDK.
- Uses local embeddings (`all-MiniLM-L6-v2`) + local ChromaDB (`chroma_db/`) to retrieve relevant chunks before asking the model.
- Currently points to `docs/CrimeAndPunishment.txt` and retrieves `n_results=5` for the question: `What does this say about Denis?`

3. `bm25/ask_bm25_rag_google.py`
- RAG with Google Gemini for generation via OpenAI SDK and in-memory BM25 lexical retrieval.
- See [bm25/README.md](./bm25/README.md) for how it works, how to run it, and the BM25 challenge.

## Shared retrieval interface

The dense baseline, BM25, and graph retrievers use the same chunk identity and
result fields so their ranked lists can be fused:

- Each chunk has a stable `chunk_id` within its corpus, its exact `text`,
  `source`, `corpus`, and `provenance`.
- Each retrieval result has `chunk_id`, `text`, `source`, numeric `score`,
  one-based `rank`, `retriever`, `corpus`, and `provenance`.
- Graph results may also include `path`; RRF results include `components`,
  which records each retriever's original rank, score, and graph path when
  supplied.

The shared chunker preserves the dense example's paragraph filtering and
800-character windows with 10% overlap. It assigns `id_0`, `id_1`, and so on,
matching the current Chroma IDs. IDs are stable for a given corpus and
chunking configuration; changing the text or chunk settings can change them.
Keep the corpus name and chunking settings consistent across retrievers before
fusing results.

The dense baseline is the existing local Chroma + `all-MiniLM-L6-v2` pipeline
in `ask_rag_google.py` (baseline commit
`f61dfa8ec83a1237f0fe4fd8ae47feaeb3d35826`). Run it with
`python ask_rag_google.py`. It remains the baseline configuration; this work
only exposes its results in the shared shape and uses the shared chunker when
building a new collection.

The V2 evaluation questions and assignment Kestrel corpus are not in this
checkout. BM25 is corpus-agnostic and can index a supplied Kestrel document,
but no Kestrel measurement or result is claimed here. The included documents
are the Crime and Punishment demo and the Vellora manual.

## BM25 and Reciprocal Rank Fusion

The reusable implementation is in `retrieval/`. BM25 uses case-folded
`\w+` tokens, positive BM25 IDF, `k1=1.5`, and `b=0.75`; exact identifiers such
as `E47` remain single tokens. The index is built in memory from the selected
document every run. It requires no additional dependency or external service,
which is appropriate for the small teaching corpora; no persistence or
incremental-update behavior is claimed.

Run lexical retrieval on the default Vellora manual without making an API call:

```powershell
python bm25/ask_bm25_rag_google.py --retrieval-only
```

Use the same retriever on the dense-baseline novel or another UTF-8 corpus:

```powershell
python bm25/ask_bm25_rag_google.py --document docs/CrimeAndPunishment.txt --question "What does this say about Denis?" --retrieval-only
```

For a corpus directory, BM25 indexes sorted `.txt` and `.md` files and gives
each chunk a source-qualified ID such as `guide.txt::0`, matching the graph
retriever's standalone chunking convention:

```powershell
python bm25/ask_bm25_rag_google.py --document path\to\kestrel\docs --corpus kestrel --question "..." --retrieval-only
```

Omit `--retrieval-only` to send the retrieved context to Gemini; that requires
`GOOGLE_API_KEY`. The default top-k is 5 and can be changed with `--top-k`.

`retrieve_with_rrf()` accepts a mapping of enabled retriever names to objects
with a `search(query, top_k=...)` method. Only retrievers present in that
mapping are called, so a disabled retriever adds neither results nor cost.
The defaults are candidate depth 20, final top-k 10, and plain (unweighted)
RRF constant 60. RRF deduplicates by `(corpus, chunk_id)`, sorts ties
deterministically, and keeps each input rank/score in `components` for
inspection. If a retriever's results do not include a corpus (the current
graph result shape), supply it explicitly, e.g.
`retrieve_with_rrf(enabled_retrievers, question, corpus="kestrel")`. Graph
paths remain available in the fused `components` and provenance. The constant
and candidate depth are defaults, not measured claims that hybrid retrieval
is better.

Run the dependency-free retrieval tests with:

```powershell
python -m unittest discover -s tests -v
```

## Project Structure

- `docs/CrimeAndPunishment.txt`: main source text used by the scripts.
- `docs/CrimeAndPunishment-shorter.txt`: shortened version of the main source text for quicker testing and rapid experimentation in Gemini online at https://gemini.google.com/app (See note below)
- `chroma_db/`: persistent local vector database (used by RAG scripts).
- `bm25/`: BM25 retrieval example and challenges.
- `retrieval/`: shared chunking, result contract, dense adapter, BM25, and RRF.
- `tests/`: dependency-free tests for retrieval and fusion.
- `concepts/`: multi-hop retrieval challenge.
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.10+ (recommended)
- API key:
   - `GOOGLE_API_KEY` from [Google AI Studio](https://aistudio.google.com/app/apikey)


## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env` and set your real keys.

## Run Each Application

With your virtual environment activated:

### 1) NoRAG + Google

```bash
python ask_norag_google.py
```

### 2) RAG + Google

```bash
python ask_rag_google.py
```

### 3) BM25 RAG + Google

```bash
python bm25/ask_bm25_rag_google.py
```

## Notes

- Both applications use the OpenAI SDK to interact with Google Gemini via Google's OpenAI-compatible endpoint.
- The RAG app splits the text into fixed-size chunks with an overlap (`CHUNK_SIZE`, `OVERLAP_PERCENT` and `MIN_PARAGRAPH_LENGTH` at the top of the script), indexes them into `chroma_db/` and reuses them across runs.
- There is one index per document (`DOCUMENT_FILE`) and chunking configuration. When no index exists for the current document and settings, the script asks before building one (this takes a little while).
- Debug prompt printing can be enabled by passing `debug=True` in the NoRAG app.
- RAG app prints retrieved context for verification.
- Streaming output is enabled by default in both applications (controlled by `STREAM_ON` variable).

- Using the shortened version of the text (`docs/CrimeAndPunishment-shorter.txt`)
	- Go online at https://gemini.google.com/app
	- At the prompt, start writing "Here is a story: ",
	- Then paste the content of docs/CrimeAndPunishment-shorter.txt into the Gemini chat app and then, at the end of the prompt, ask "What can you tell me about Denis Amselemovitch?"
	- Observe the response from Gemini.

# Free books from Project Gutenberg
- https://www.gutenberg.org/
# Crime and Punishment by Fyodor Dostoevsky
- https://www.gutenberg.org/ebooks/2554
# Huckleberry Finn by Mark Twain
- https://www.gutenberg.org/ebooks/76
