import os
import unittest
from pathlib import Path

import tracing
from tracing import Span, Trace, compute_cost, compute_fingerprint, get_metrics, get_trace, list_traces, record_trace


class ObservabilityTests(unittest.TestCase):
    def test_cost_calculation(self):
        # 1M in = $0.075, 1M out = $0.30
        cost = compute_cost(1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.375, places=4)

        # 1000 in, 100 out
        cost_small = compute_cost(1000, 100)
        self.assertGreater(cost_small, 0.0)
        self.assertLess(cost_small, 0.001)

    def test_fingerprint_deterministic(self):
        cfg1 = {"top_k": 5, "retrievers": ["dense", "bm25"]}
        cfg2 = {"retrievers": ["dense", "bm25"], "top_k": 5}
        self.assertEqual(compute_fingerprint(cfg1), compute_fingerprint(cfg2))

    def test_trace_lifecycle_and_spans(self):
        trace = Trace(name="test.request", fingerprint="abc12345")
        with trace.span("test.retrieve") as s:
            s.set_attribute("candidates", 10)

        with trace.span("test.generate") as s:
            s.set_attribute("tokens_in", 200)
            s.set_attribute("tokens_out", 50)

        trace.finish(outcome="answered", status="ok")

        data = trace.to_dict()
        self.assertEqual(data["name"], "test.request")
        self.assertEqual(data["outcome"], "answered")
        self.assertEqual(data["status"], "ok")
        self.assertEqual(len(data["spans"]), 2)
        self.assertEqual(data["spans"][0]["name"], "test.retrieve")
        self.assertEqual(data["spans"][1]["name"], "test.generate")
        self.assertEqual(data["attributes"]["tokens_in"], 200)
        self.assertEqual(data["attributes"]["tokens_out"], 50)
        self.assertGreater(data["attributes"]["cost_usd"], 0.0)

    def test_record_and_get_trace(self):
        trace = Trace(name="unit_test_trace")
        with trace.span("step_1") as s:
            s.set_attribute("key", "val")
        trace.finish(outcome="answered")
        record_trace(trace)

        fetched = get_trace(trace.trace_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["trace_id"], trace.trace_id)
        self.assertEqual(len(fetched["spans"]), 1)

    def test_metrics_aggregation(self):
        m = get_metrics()
        self.assertIn("total_requests", m)
        self.assertIn("outcomes", m)
        self.assertIn("latency", m)
        self.assertIn("overall_p50", m["latency"])
        self.assertIn("overall_p95", m["latency"])
        self.assertIn("cost", m)


if __name__ == "__main__":
    unittest.main()
