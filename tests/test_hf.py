"""Tests for the HuggingFace ingestion helper.

These tests mock ``datasets.load_dataset`` so that no network access is
required. Each test verifies that the helper's argument-passthrough,
adapter dispatch, and bounding logic match the documented contract.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from procgrep.hf import from_hf
from procgrep.types import ATOM_EDIT, ATOM_RUN_TEST, ATOM_THINK


class _FakeDataset:
    """Minimal stand-in for a ``datasets.Dataset`` used in unit tests.

    Supports the operations ``from_hf`` performs: ``len``, ``select``,
    iteration, and ``take`` (the streaming-mode method).
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Any:
        return iter(self._rows)

    def select(self, indices: range) -> _FakeDataset:
        return _FakeDataset([self._rows[i] for i in indices])

    def take(self, n: int) -> _FakeDataset:
        return _FakeDataset(self._rows[:n])


def _swe_smith_row(
    traj_id: str, action: str, *, model: str = "claude-3-7-sonnet"
) -> dict[str, Any]:
    """Build a minimal SWE-smith-shaped HF row for tests."""
    messages = [
        {"role": "system", "content": "You are..."},
        {"role": "assistant", "thought": "Thinking", "action": action},
    ]
    return {
        "messages": json.dumps(messages),
        "traj_id": traj_id,
        "model": model,
        "resolved": True,
        "instance_id": f"inst_{traj_id}",
    }


@pytest.fixture
def fake_swe_smith_dataset() -> _FakeDataset:
    """A small SWE-smith-shaped dataset with three rows."""
    return _FakeDataset(
        [
            _swe_smith_row("t1", "edit"),
            _swe_smith_row("t2", "pytest"),
            _swe_smith_row("t3", "submit"),
        ]
    )


def test_calls_load_dataset_with_expected_args(fake_swe_smith_dataset: _FakeDataset) -> None:
    """from_hf forwards dataset name, config, split, streaming, and revision."""
    with patch("datasets.load_dataset", return_value=fake_swe_smith_dataset) as mock_load:
        from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            config_name="default",
            streaming=False,
            revision="main",
            trace_id_field="traj_id",
            agent_field="model",
        )
    mock_load.assert_called_once_with(
        "SWE-bench/SWE-smith-trajectories",
        name="default",
        split="tool",
        streaming=False,
        revision="main",
    )


def test_canonicalizes_rows_to_traces(fake_swe_smith_dataset: _FakeDataset) -> None:
    """The returned objects are canonical Traces with atoms from the adapter."""
    with patch("datasets.load_dataset", return_value=fake_swe_smith_dataset):
        traces = from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            trace_id_field="traj_id",
            agent_field="model",
        )
    assert len(traces) == 3
    assert traces[0].trace_id == "t1"
    assert traces[0].agent == "claude-3-7-sonnet"
    # Each row had one thought + one action → ATOM_THINK + mapped atom.
    assert traces[0].atoms == [ATOM_THINK, ATOM_EDIT]
    assert traces[1].atoms == [ATOM_THINK, ATOM_RUN_TEST]


def test_limit_eager_uses_select(fake_swe_smith_dataset: _FakeDataset) -> None:
    """In eager mode (streaming=False), limit selects via dataset.select."""
    with patch("datasets.load_dataset", return_value=fake_swe_smith_dataset):
        traces = from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            trace_id_field="traj_id",
            agent_field="model",
            limit=2,
        )
    assert len(traces) == 2
    assert [t.trace_id for t in traces] == ["t1", "t2"]


def test_limit_eager_handles_short_corpus(fake_swe_smith_dataset: _FakeDataset) -> None:
    """If limit exceeds dataset length, all rows are returned without error."""
    with patch("datasets.load_dataset", return_value=fake_swe_smith_dataset):
        traces = from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            trace_id_field="traj_id",
            agent_field="model",
            limit=100,
        )
    assert len(traces) == 3


def test_limit_streaming_uses_take(fake_swe_smith_dataset: _FakeDataset) -> None:
    """In streaming mode, limit uses dataset.take rather than select."""
    take_spy = MagicMock(wraps=fake_swe_smith_dataset.take)
    fake_swe_smith_dataset.take = take_spy  # type: ignore[method-assign]
    with patch("datasets.load_dataset", return_value=fake_swe_smith_dataset):
        traces = from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            streaming=True,
            trace_id_field="traj_id",
            agent_field="model",
            limit=2,
        )
    take_spy.assert_called_once_with(2)
    assert len(traces) == 2


def test_no_limit_returns_full_corpus(fake_swe_smith_dataset: _FakeDataset) -> None:
    """When limit is None, all rows pass through (eager mode)."""
    with patch("datasets.load_dataset", return_value=fake_swe_smith_dataset):
        traces = from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            trace_id_field="traj_id",
            agent_field="model",
        )
    assert len(traces) == 3


def test_passes_through_group_field(fake_swe_smith_dataset: _FakeDataset) -> None:
    """The group_field parameter reaches canonicalize and lands on each Trace."""
    # Inject a "group" field into each row.
    rows = [
        {**row, "group_label": "case_study_1"}
        for row in [
            _swe_smith_row("t1", "edit"),
            _swe_smith_row("t2", "edit"),
        ]
    ]
    dataset = _FakeDataset(rows)
    with patch("datasets.load_dataset", return_value=dataset):
        traces = from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            trace_id_field="traj_id",
            agent_field="model",
            group_field="group_label",
        )
    assert all(t.group == "case_study_1" for t in traces)


def test_unknown_adapter_raises(fake_swe_smith_dataset: _FakeDataset) -> None:
    """An unregistered adapter name propagates a KeyError from canonicalize."""
    with (
        patch("datasets.load_dataset", return_value=fake_swe_smith_dataset),
        pytest.raises(KeyError, match="no adapter named"),
    ):
        from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="nonexistent-adapter",
            split="tool",
            trace_id_field="traj_id",
            agent_field="model",
        )


def test_import_error_when_datasets_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the ``datasets`` package is unavailable, from_hf raises ImportError."""
    # Remove already-imported datasets module so the lazy import fails.
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match="datasets"):
        from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            trace_id_field="traj_id",
            agent_field="model",
        )


def test_metadata_carries_non_canonical_fields(
    fake_swe_smith_dataset: _FakeDataset,
) -> None:
    """Fields not consumed by trace_id/agent/group land in Trace.metadata."""
    with patch("datasets.load_dataset", return_value=fake_swe_smith_dataset):
        traces = from_hf(
            "SWE-bench/SWE-smith-trajectories",
            adapter="swe-smith",
            split="tool",
            trace_id_field="traj_id",
            agent_field="model",
        )
    # `resolved` is a non-canonical field; canonicalize stores it as metadata.
    assert traces[0].metadata.get("resolved") is True
    assert "instance_id" in traces[0].metadata
