"""Tests for `procgrep.types`."""

from __future__ import annotations

from procgrep.types import CANONICAL_ATOMS, Trace


def test_canonical_alphabet_is_nonempty() -> None:
    assert len(CANONICAL_ATOMS) >= 10
    assert "edit" in CANONICAL_ATOMS
    assert "run_test" in CANONICAL_ATOMS


def test_trace_grouping_falls_back_to_agent_when_group_is_none() -> None:
    trace = Trace(trace_id="t1", agent="alpha", atoms=["edit"])
    assert trace.grouping() == "alpha"


def test_trace_grouping_uses_group_when_set() -> None:
    trace = Trace(trace_id="t1", agent="alpha", atoms=["edit"], group="cell-A")
    assert trace.grouping() == "cell-A"


def test_trace_equality_by_value() -> None:
    a = Trace(trace_id="t1", agent="alpha", atoms=["edit"])
    b = Trace(trace_id="t1", agent="alpha", atoms=["edit"])
    assert a == b


def test_trace_inequality_when_atoms_differ() -> None:
    a = Trace(trace_id="t1", agent="alpha", atoms=["edit"])
    b = Trace(trace_id="t1", agent="alpha", atoms=["edit", "run_test"])
    assert a != b
