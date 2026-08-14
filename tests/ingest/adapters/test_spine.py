"""Tests for the precomputed-spine adapter."""

from __future__ import annotations

from procgrep.canonicalize import canonicalize, get_adapter
from procgrep.ingest.adapters.spine import spine_adapter


def test_splits_space_joined_atoms() -> None:
    record = {"spine": "think edit run_test  submit"}
    assert spine_adapter(record) == ["think", "edit", "run_test", "submit"]


def test_non_string_or_missing_spine_is_empty() -> None:
    assert spine_adapter({}) == []
    assert spine_adapter({"spine": None}) == []
    assert spine_adapter({"spine": ["edit"]}) == []


def test_registered_and_canonicalizes() -> None:
    assert get_adapter("spine") is spine_adapter
    traces = canonicalize(
        [{"trace_id": "t1", "agent": "a", "dataset": "d1", "spine": "edit run_test"}],
        adapter="spine",
    )
    assert traces[0].atoms == ["edit", "run_test"]
    # non-identity columns survive as metadata, so they can drive grouping later
    assert traces[0].metadata["dataset"] == "d1"
