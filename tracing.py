import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from rag_core import BASE_DIR

TRACES_FILE = BASE_DIR / "data" / "traces.jsonl"
TRACES_LOCK = threading.Lock()

RETENTION_DAYS = int(os.getenv("TRACE_RETENTION_DAYS", "30"))
PRUNE_EVERY = 100
_writes = 0

COST_PER_INPUT_TOKEN = float(os.getenv("PRICE_INPUT_PER_M", "0.075")) / 1_000_000
COST_PER_OUTPUT_TOKEN = float(os.getenv("PRICE_OUTPUT_PER_M", "0.30")) / 1_000_000


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return round((input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN), 6)


def compute_fingerprint(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


class Span:
    def __init__(self, name: str, span_id: str | None = None, parent_span_id: str | None = None):
        self.span_id = span_id or uuid.uuid4().hex[:8]
        self.parent_span_id = parent_span_id
        self.name = name
        self.start_time = time.time()
        self.end_time = 0.0
        self.duration_ms = 0
        self.status = "ok"
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def finish(self, status: str | None = None):
        if self.end_time == 0.0:
            self.end_time = time.time()
            self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        if status:
            self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
        }


class SpanContext:
    def __init__(self, trace: "Trace", span: Span):
        self.trace = trace
        self.span = span

    def __enter__(self) -> Span:
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.finish("error")
            self.span.set_attribute("error", str(exc_val))
        else:
            self.span.finish("ok")
        self.trace.current_span_id = self.span.parent_span_id


class Trace:
    def __init__(self, name: str, trace_id: str | None = None, fingerprint: str = "", attributes: dict[str, Any] | None = None):
        self.trace_id = trace_id or uuid.uuid4().hex[:8]
        self.name = name
        self.fingerprint = fingerprint
        self.start_time = time.time()
        self.end_time = 0.0
        self.duration_ms = 0
        self.status = "ok"
        self.outcome = "unknown"
        self.attributes = attributes or {}
        self.spans: list[Span] = []
        self.current_span_id: str | None = None

    def span(self, name: str) -> SpanContext:
        new_span = Span(name=name, parent_span_id=self.current_span_id)
        self.spans.append(new_span)
        self.current_span_id = new_span.span_id
        return SpanContext(self, new_span)

    def finish(self, outcome: str = "answered", status: str = "ok"):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.outcome = outcome
        self.status = status
        # Calculate aggregate token & cost metrics across all spans
        total_in = 0
        total_out = 0
        for s in self.spans:
            s.finish()
            total_in += s.attributes.get("tokens_in", 0)
            total_out += s.attributes.get("tokens_out", 0)
        self.attributes["tokens_in"] = total_in
        self.attributes["tokens_out"] = total_out
        self.attributes["tokens_total"] = total_in + total_out
        self.attributes["cost_usd"] = compute_cost(total_in, total_out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "fingerprint": self.fingerprint,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.start_time)),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "outcome": self.outcome,
            "attributes": self.attributes,
            "spans": [s.to_dict() for s in self.spans],
        }


def prune_traces(days: int | None = None) -> int:
    days = RETENTION_DAYS if days is None else days
    if not TRACES_FILE.exists():
        return 0
    limit = time.time() - days * 86400
    with TRACES_LOCK:
        lines = [line for line in TRACES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept = []
        for line in lines:
            try:
                started = float(json.loads(line).get("start_time", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if started >= limit:
                kept.append(line)
        removed = len(lines) - len(kept)
        if removed:
            TRACES_FILE.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def record_trace(trace: Trace):
    global _writes
    TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = trace.to_dict()
    with TRACES_LOCK:
        with TRACES_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _writes += 1
        due = _writes % PRUNE_EVERY == 0
    if due:
        prune_traces()


def get_trace(trace_id: str) -> dict[str, Any] | None:
    if not TRACES_FILE.exists():
        return None
    with TRACES_LOCK:
        lines = TRACES_FILE.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if data.get("trace_id") == trace_id:
                return data
        except json.JSONDecodeError:
            continue
    return None


def list_traces(limit: int = 50) -> list[dict[str, Any]]:
    if not TRACES_FILE.exists():
        return []
    traces = []
    with TRACES_LOCK:
        lines = TRACES_FILE.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            traces.append(json.loads(line))
            if len(traces) >= limit:
                break
        except json.JSONDecodeError:
            continue
    return traces


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    d = k - f
    return round(sorted_v[f] + d * (sorted_v[c] - sorted_v[f]), 2)


def get_metrics() -> dict[str, Any]:
    if not TRACES_FILE.exists():
        return {
            "total_requests": 0,
            "outcomes": {"answered": 0, "abstained": 0, "error": 0, "other": 0},
            "latency": {"overall_p50": 0.0, "overall_p95": 0.0, "by_step": {}},
            "tokens": {"total_in": 0, "total_out": 0, "total": 0},
            "cost": {"total_usd": 0.0},
            "daily": {},
        }

    with TRACES_LOCK:
        lines = TRACES_FILE.read_text(encoding="utf-8").splitlines()

    requests_count = 0
    indexing_count = 0
    outcomes = {"answered": 0, "abstained": 0, "error": 0, "other": 0}
    latencies: list[float] = []
    step_latencies: dict[str, list[float]] = {}
    tokens_in = 0
    tokens_out = 0
    daily_stats: dict[str, dict[str, Any]] = {}

    for line in lines:
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue

        day = (t.get("timestamp") or "")[:10] or "unknown"
        if day not in daily_stats:
            daily_stats[day] = {"requests": 0, "cost_usd": 0.0, "tokens": 0}

        if t.get("name") == "indexing.run":
            indexing_count += 1
            t_cost = t.get("attributes", {}).get("cost_usd", 0.0)
            daily_stats[day]["cost_usd"] += t_cost
            continue

        requests_count += 1
        daily_stats[day]["requests"] += 1

        outcome = t.get("outcome", "other")
        if outcome in outcomes:
            outcomes[outcome] += 1
        else:
            outcomes["other"] += 1

        dur = float(t.get("duration_ms", 0.0))
        latencies.append(dur)

        tin = t.get("attributes", {}).get("tokens_in", 0)
        tout = t.get("attributes", {}).get("tokens_out", 0)
        t_cost = t.get("attributes", {}).get("cost_usd", 0.0)
        tokens_in += tin
        tokens_out += tout
        daily_stats[day]["tokens"] += (tin + tout)
        daily_stats[day]["cost_usd"] += t_cost

        for s in t.get("spans", []):
            s_name = s.get("name", "unknown")
            s_dur = float(s.get("duration_ms", 0.0))
            if s_name not in step_latencies:
                step_latencies[s_name] = []
            step_latencies[s_name].append(s_dur)

    by_step_metrics = {}
    for s_name, s_durs in step_latencies.items():
        by_step_metrics[s_name] = {
            "p50": percentile(s_durs, 50),
            "p95": percentile(s_durs, 95),
            "count": len(s_durs),
        }

    for day in daily_stats:
        daily_stats[day]["cost_usd"] = round(daily_stats[day]["cost_usd"], 6)

    total_cost = compute_cost(tokens_in, tokens_out)

    return {
        "total_requests": requests_count,
        "total_indexing_runs": indexing_count,
        "outcomes": outcomes,
        "latency": {
            "overall_p50": percentile(latencies, 50),
            "overall_p95": percentile(latencies, 95),
            "by_step": by_step_metrics,
        },
        "tokens": {
            "total_in": tokens_in,
            "total_out": tokens_out,
            "total": tokens_in + tokens_out,
        },
        "cost": {
            "total_usd": total_cost,
        },
        "daily": daily_stats,
    }
