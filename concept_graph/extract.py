"""LLM triple extraction, one call per chunk, with schema validation and provenance checks.

Malformed output policy (never silently accept):
1. The reply is not valid JSON / not {"triples": [...]}: retry ONCE, then log the chunk as failed.
2. One triple breaks the schema (unknown type, empty name...): drop that triple, log it, keep the others.
3. The evidence quote is not found in the chunk text: drop it, log it as "evidence_not_in_chunk".
Every rejected item goes to rejected.jsonl with its reason and chunk ID.

Results are cached by hash(model + prompt version + chunk text), so a rebuild with the same
corpus and settings makes no new LLM calls and gives the same graph.
"""
import hashlib
import json
import os
import re
import time

from pydantic import ValidationError

from .schema import SCHEMA_DESCRIPTION, Triple

PROMPT_VERSION = "v1"
DEFAULT_MODEL = os.getenv("GRAPH_MODEL", "gemini-3.5-flash-lite")

PROMPT = """You extract a knowledge graph from ONE passage of a document.
Return JSON only: {{"triples": [{{"subject": {{"name": ..., "type": ...}}, "relation": ..., "object": {{"name": ..., "type": ...}}, "evidence": ...}}]}}

{schema}

Rules:
- Only facts stated in the passage. No outside knowledge, no guesses.
- "evidence" is copied word for word from the passage (one sentence or part of one).
- Use the most specific relation type. RELATED_TO only if nothing else fits.
- Use names as written in the passage. If the passage says two names are the same thing
  ("X is called Y", "X replaced Y and old mentions of X mean Y"), add a SAME_AS or REPLACES triple.
- Error codes keep their exact form (KST-5031). Channels keep their # (#oncall-messaging).
- At most 15 triples. If the passage states no relation between entities, return {{"triples": []}}.

Passage (source: {source}):
\"\"\"
{text}
\"\"\"
"""


def _norm_ws(s):
    return " ".join(s.split()).lower()


def evidence_in_chunk(evidence, chunk_text):
    """Whitespace- and case-insensitive substring test, ignoring markdown ** and quotes."""
    clean = lambda s: _norm_ws(re.sub(r"[*`\"“”‘’]", "", s))
    return clean(evidence) in clean(chunk_text)


class GeminiLLM:
    """OpenAI-compatible Gemini client (same endpoint and key as ask_rag_google.py)."""

    def __init__(self, model=DEFAULT_MODEL):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set (see .env.example)")
        self.client = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.model = model

    def __call__(self, prompt):
        from openai import RateLimitError
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                break
            except RateLimitError:
                # Free tier quota (requests per minute): wait and try again. Not counted as an LLM call.
                if attempt == 5:
                    raise
                wait = 10 * (attempt + 1)
                print(f"   429 rate limit, waiting {wait}s (or set GRAPH_MODEL to another model)")
                time.sleep(wait)
        usage = resp.usage
        return resp.choices[0].message.content, {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }


def _parse_reply(reply):
    text = reply.strip()
    m = re.search(r"\{.*\}", text, re.S)  # tolerate ```json fences
    if not m:
        raise ValueError("no JSON object in reply")
    data = json.loads(m.group(0))
    if not isinstance(data, dict) or not isinstance(data.get("triples"), list):
        raise ValueError('reply is not {"triples": [...]}')
    return data["triples"]


class Extractor:
    def __init__(self, llm, model_name, cache_path=None, max_retries=1):
        self.llm = llm
        self.model_name = model_name
        self.max_retries = max_retries
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row["key"]] = row["raw_triples"]
        self.stats = {"llm_calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "llm_seconds": 0.0, "failed_chunks": 0, "triples_raw": 0, "triples_valid": 0}
        self.rejected = []

    def _key(self, chunk):
        h = hashlib.sha256(f"{self.model_name}|{PROMPT_VERSION}|{chunk['text']}".encode()).hexdigest()
        return h[:24]

    def _call(self, chunk):
        prompt = PROMPT.format(schema=SCHEMA_DESCRIPTION, source=chunk["source"], text=chunk["text"])
        last_error = None
        for _ in range(1 + self.max_retries):
            t0 = time.perf_counter()
            reply, usage = self.llm(prompt)
            self.stats["llm_seconds"] += time.perf_counter() - t0
            self.stats["llm_calls"] += 1
            self.stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.stats["completion_tokens"] += usage.get("completion_tokens", 0)
            try:
                return _parse_reply(reply)
            except (ValueError, json.JSONDecodeError) as e:
                last_error = f"{type(e).__name__}: {e}"
                self.rejected.append({"chunk_id": chunk["chunk_id"], "reason": "malformed_reply",
                                      "detail": last_error, "reply": reply[:500]})
        return None

    def extract(self, chunk):
        """Returns the list of validated Triple objects for one chunk."""
        key = self._key(chunk)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            raw = self.cache[key]
        else:
            raw = self._call(chunk)
            if raw is None:
                self.stats["failed_chunks"] += 1
                return []
            self.cache[key] = raw
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "chunk_id": chunk["chunk_id"], "raw_triples": raw},
                                       ensure_ascii=False) + "\n")
        valid = []
        for item in raw:
            self.stats["triples_raw"] += 1
            try:
                t = Triple.model_validate(item)
            except ValidationError as e:
                self.rejected.append({"chunk_id": chunk["chunk_id"], "reason": "schema",
                                      "detail": e.errors(include_url=False, include_input=False)[:3],
                                      "item": item})
                continue
            if not evidence_in_chunk(t.evidence, chunk["text"]):
                self.rejected.append({"chunk_id": chunk["chunk_id"], "reason": "evidence_not_in_chunk",
                                      "item": item})
                continue
            valid.append(t)
        self.stats["triples_valid"] += len(valid)
        return valid
