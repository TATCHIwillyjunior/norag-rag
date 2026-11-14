# Concept graph retriever (V3, Kestrel)

Owner: Keylan (`feat_kelyan`). Tasks K1 to K8 of `TODO.md`.

The graph retriever finds chunks that are linked to the question through facts, not through
similar words. It is meant for `two_hop` questions, where the answer is spread over two passages
and the second passage shares no words with the question (q08: KST-5031 → postroom → Messaging).

## Run it

```powershell
pip install -r concept_graph/requirements.txt   # needs GOOGLE_API_KEY in .env, like ask_rag_google.py

# 1. Build (one LLM call per chunk; a rebuild reuses extract_cache.jsonl and makes no new calls)
python -m concept_graph.build --docs path\to\kestrel\docs --out graph_out\kestrel
#    or, once the app exports its chunks (same chunk IDs as dense and BM25, needed for RRF):
python -m concept_graph.build --chunks graph_out\kestrel_chunks.jsonl --out graph_out\kestrel

# 2. Ask
python -m concept_graph.query --graph graph_out\kestrel "Error KST-5031 happened on the last two scheduled reports. Which team should support contact, and where?"

# 3. Quality check: sample 10 random triples, grade them by hand, then compute the totals
python -m concept_graph.triples sample  --graph graph_out\kestrel
python -m concept_graph.triples summary --graph graph_out\kestrel

# Offline tests (fake LLM, no key needed)
python -m pytest tests/test_concept_graph.py -q
```

Settings: `GRAPH_MODEL` (default `gemini-3.5-flash-lite`), `--chunk-size`/`--overlap` (default 800 / 10 %,
same as `ask_rag_google.py`), `--hops` at query time (default 2).

## Design choices (K2)

| Option | What it does | Why not / why |
|---|---|---|
| Microsoft GraphRAG | LLM extraction + community detection + LLM summaries per community | Built for "global" questions over big corpora. Many LLM calls per build and per query. Overkill for 5 short documents and 2-hop lookups. |
| LightRAG | LLM extraction + dual (entity / theme) retrieval, keyword LLM call per query | Closer to what we need, but adds an LLM call per question and hides the steps we must explain and measure. |
| LangChain `LLMGraphTransformer` / LlamaIndex `KnowledgeGraphIndex` | Library extraction into a graph store | Fast to start, but schema validation, provenance and resolution would be library internals; adds a heavy dependency. |
| **Chosen: small own pipeline** | Pydantic-typed extraction per chunk, exact-quote provenance, rule-based resolution, BFS over 2 hops, no LLM at query time | Every step is visible and measurable, as the assignment asks. Zero query-time cost. ~400 lines. |

Cost of the choice: entity linking at query time is lexical (names must appear in the question),
so a question that only paraphrases an entity finds nothing in the graph. Dense retrieval covers that case in the fusion.

## Schema (K3)

Fixed types, defined in `schema.py` and validated with Pydantic:

- **Entities:** Service, Team, Person, Channel, ErrorCode, Plan, Feature, Policy, Document, Place, Object, Organization.
- **Relations:** OWNS, LEADS, MEMBER_OF, CONTACT_VIA, RAISES, ESCALATE_TO, DEPENDS_ON, SENDS, REPLACES,
  INCLUDES, APPLIES_TO, DEFINED_IN, LOCATED_IN, MOVED_TO, HAS, LENT_TO, SAME_AS, RELATED_TO (generic fallback, counted separately).

Each triple must carry an `evidence` quote. Malformed output is never silently accepted:

| Problem | Action |
|---|---|
| Reply is not JSON `{"triples": [...]}` | retry once, then the chunk is logged as failed |
| A triple breaks the schema (unknown type or relation, empty name) | that triple is dropped and logged, the others are kept |
| The evidence quote is not in the chunk | dropped, logged as `evidence_not_in_chunk` (would be "unsupported") |

All rejections are in `rejected.jsonl`, with counts in `build_stats.json`.

## Provenance (K4)

- **Nodes** keep `chunk_ids` (every chunk that mentions them) and `aliases` (every original name).
- **Edges** keep `chunk_id` and the `evidence` quote.

The retriever only returns chunks that exist in `chunks.jsonl`, so it never makes up text.
`build_stats.json` records:
- the model and prompt version;
- LLM calls, cache hits, prompt, completion and total tokens;
- LLM time and total build time;
- failed chunks, raw, valid and rejected triples;
- entities and relations, by type;
- merges.

## Entity resolution (K5)

1. **Normalization.** Names are lower-cased, with accents, `'s`, articles and a trailing " service" or " team" removed. Names that end up with the same key **and the same type** become one node: "Postroom", "postroom" and "the postroom service" all become `Service:postroom`.
2. **Same name, different type is not merged.** "Messaging" the team and "messaging" the feature stay apart, and this is logged as `not_merged_type_conflict`.
3. **Explicit aliases.** When the text says two names are one thing ("The old bookbinder of Stolyarny Lane … called Yakov Glinkin"), the LLM returns a SAME_AS triple and the two nodes are merged. The chunk that states the alias is kept on the merged node, so retrieval returns it as the link between the two hops.

Everything is logged in `merges.jsonl`. **For the report**, pick from that log one correct merge and one wrong or missed merge.
Likely candidates on Kestrel:
- **Correct:** `postroom` seen with different spellings.
- **Missed:** `mailman` vs `postroom`. The catalog says postroom *replaced* mailman, so it is a REPLACES edge and not a merge. A question about a "mailman alert" still reaches postroom in 1 hop through that edge.
- **Type conflict:** "Platform" the team vs "Platform on-call".

Check the real log before writing these up. They depend on what the model extracted.

## Retrieval (K6)

1. **Find the question's entities.** A node matches when one of its names appears as whole words in the question. No LLM call is made.
2. **Walk the graph.** Breadth-first search from those nodes, both edge directions, up to 2 hops. An edge first reached at hop *h* weighs 1/*h*.
3. **Skip hubs.** A node with more than 20 edges reached during the walk is not expanded further.
4. **Score the chunks.** A chunk's score is the sum of the weights of the edges and alias links that cite it. Ties are sorted by chunk ID, so the ranking is deterministic.
5. **No match:** an empty list, with `reason: "no_entity_match"` or `"no_path"`.

## Integration handoff (K8)

```python
from concept_graph import GraphRetriever
graph = GraphRetriever.load("graph_out/kestrel/graph.json", "graph_out/kestrel/chunks.jsonl")
hits = graph.search(question, top_k=10)           # list for RRF
full = graph.explain(question, top_k=10)          # + "entities" and "reason", for the UI
```

Each hit has the shared shape `{"chunk_id", "text", "source", "score", "rank", "retriever": "graph", "path"}`.

`path` lists what led to the chunk:
- an edge: `{"hop", "source", "relation", "target", "evidence"}`;
- an alias link: `{"hop", "alias_of"}`.

The UI can show `entities` as the question's entities and `path` as the edges walked, under each reference.

**RRF needs the same chunk IDs in every list.** Build the graph with `--chunks` from the app's own chunks. The `--docs` mode uses IDs like `service-catalog.md::3`, which only match if the app uses that format too.

## Known failures and limits

- **Paraphrased entities are missed.** "the e-mail service" does not link to postroom unless the LLM saw that name. Dense retrieval covers this case.
- **Facts split across two chunks are lost.** A fact that is cut across a chunk boundary is in neither chunk's extraction. The 10 % overlap reduces this but doesn't remove it.
- **Table rows depend on the model.** The service-catalog team table (lead, channel, hours) is extracted as LEADS and CONTACT_VIA edges only if the model reads the table rows. Check `relations_by_type`.
- **Hub nodes add noise.** On a 2-hop walk, a node with many edges (Platform, Kestrel) would pull in many chunks. A hub reached during the walk (more than `max_degree` = 20 edges) is not expanded; a hub named in the question still is. Watch the case where graph chunks push a good dense chunk out of the top-k after fusion.
- **Free-tier rate limits.** Gemini can return 429 errors. The client waits and retries, and these waits are not counted as LLM calls.
