import json
import tempfile
import time
import unittest
from pathlib import Path

import tracing
from tracing import Trace, record_trace


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_file = tracing.TRACES_FILE
        tracing.TRACES_FILE = Path(self.tmp.name) / "traces.jsonl"

    def tearDown(self):
        tracing.TRACES_FILE = self.old_file
        self.tmp.cleanup()

    def add(self, trace_id, age_days):
        t = Trace(name="test.request", trace_id=trace_id)
        t.start_time = time.time() - age_days * 86400
        t.finish()
        record_trace(t)

    def ids(self):
        return [json.loads(line)["trace_id"] for line in tracing.TRACES_FILE.read_text(encoding="utf-8").splitlines()]

    def test_old_traces_are_deleted_and_recent_kept(self):
        self.add("old1", 45)
        self.add("old2", 31)
        self.add("new1", 5)
        self.assertEqual(tracing.prune_traces(30), 2)
        self.assertEqual(self.ids(), ["new1"])

    def test_nothing_to_delete(self):
        self.add("new1", 1)
        self.assertEqual(tracing.prune_traces(30), 0)
        self.assertEqual(self.ids(), ["new1"])

    def test_missing_file_is_fine(self):
        self.assertEqual(tracing.prune_traces(30), 0)

    def test_prune_runs_by_itself_after_many_writes(self):
        self.add("old", 90)
        for i in range(tracing.PRUNE_EVERY):
            self.add(f"n{i}", 0)
        self.assertNotIn("old", self.ids())


if __name__ == "__main__":
    unittest.main()
