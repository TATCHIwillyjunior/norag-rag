import pytest

import tracing


@pytest.fixture(autouse=True)
def isolated_trace_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACES_FILE", tmp_path / "traces.jsonl")
