"""Tests for the bdtrace standardized-record adapter."""

from __future__ import annotations

from procgrep.canonicalize import canonicalize, get_adapter
from procgrep.ingest.adapters.bdtrace import bdtrace_adapter


def _event(type_: str, **details: str) -> dict:
    return {"type": type_, "timestamp": "2026-09-02T00:00:00Z", "details": details}


def test_maps_the_bdtrace_taxonomy() -> None:
    record = {
        "instance_id": "claude-abc",
        "events": [
            _event("prompt", text="fix the parser"),
            _event("read", file_path="a.py"),
            _event("edit", file_path="a.py"),
            _event("search", tool="Grep"),
            _event("test", command="pytest -q"),
        ],
    }
    assert bdtrace_adapter(record) == ["prompt_ai", "read_file", "edit", "search_repo", "run_test"]


def test_run_splits_version_control_from_execution() -> None:
    record = {
        "events": [
            _event("run", tool="Bash", command="git commit -m x"),
            _event("run", tool="Bash", command="python build.py"),
            _event("run", tool="Bash"),
        ]
    }
    assert bdtrace_adapter(record) == ["version_control", "run_code", "run_code"]


def test_legacy_code_change_and_unknown_types() -> None:
    record = {"events": [_event("code_change", file_path="a.py"), _event("mystery")]}
    assert bdtrace_adapter(record) == ["edit", "other"]


def test_malformed_records_are_empty_not_fatal() -> None:
    assert bdtrace_adapter({}) == []
    assert bdtrace_adapter({"events": None}) == []
    assert bdtrace_adapter({"events": ["not a mapping"]}) == []


def test_registered_and_canonicalizes_a_bdtrace_export() -> None:
    assert get_adapter("bdtrace") is bdtrace_adapter
    traces = canonicalize(
        [{"instance_id": "claude-abc", "agent": "claude", "events": [_event("edit"), _event("test")]}],
        adapter="bdtrace",
        trace_id_field="instance_id",
    )
    assert traces[0].atoms == ["edit", "run_test"]
