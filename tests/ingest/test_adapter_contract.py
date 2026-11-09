"""A new trace format must reuse the existing measurement pipeline."""

from __future__ import annotations

from importlib import import_module

import pytest

from procgrep import Trace, canonicalize, encode, fit_bpe, jsd
from procgrep.canonicalize import get_adapter, make_action_adapter, register_adapter
from procgrep.types import TraceAdapter


@pytest.fixture
def event_adapter() -> TraceAdapter:
    """An evaluation-only format with operation names unlike the built-ins."""
    return make_action_adapter(
        actions_path="events",
        action_field="op",
        atom_map={
            "OPEN": "read_file",
            "PATCH": "edit",
            "CHECK": "run_test",
            "DONE": "submit",
        },
    )


def _raw_trace(operations: list[str]) -> dict[str, object]:
    return {
        "trace_id": "example",
        "agent": "example-agent",
        "group": "control",
        "events": [{"op": operation} for operation in operations],
    }


def test_new_format_preserves_actions_identity_and_measurements(
    event_adapter: TraceAdapter,
) -> None:
    expected = Trace(
        trace_id="example",
        agent="example-agent",
        group="control",
        atoms=["read_file", "edit", "run_test", "edit", "run_test", "submit"],
    )
    raw = _raw_trace(["OPEN", "PATCH", "CHECK", "PATCH", "CHECK", "DONE"])
    (actual,) = canonicalize([raw], adapter=event_adapter)
    assert actual.atoms == expected.atoms
    assert (actual.trace_id, actual.agent, actual.group) == (
        expected.trace_id,
        expected.agent,
        expected.group,
    )

    vocab = fit_bpe([expected.atoms], vocab_size=5)
    assert vocab.merges  # Exercise learned procedures, not only individual atoms.
    expected_fp, actual_fp = encode([expected, actual], vocab=vocab)
    assert actual_fp == expected_fp
    assert jsd(expected_fp.distribution(), actual_fp.distribution()) == 0.0

    (different,) = canonicalize([_raw_trace(["OPEN"])], adapter=event_adapter)
    (different_fp,) = encode([different], vocab=vocab)
    assert jsd(expected_fp.distribution(), different_fp.distribution()) > 0.0


def test_new_format_can_register_without_changing_builtin_adapters(
    event_adapter: TraceAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = import_module("procgrep.canonicalize")
    monkeypatch.setattr(module, "_ADAPTERS", module._ADAPTERS.copy())
    original = get_adapter("swe-agent")
    register_adapter("contract-events", event_adapter)

    (trace,) = canonicalize([_raw_trace(["OPEN", "PATCH"])], adapter="contract-events")
    assert trace.atoms == ["read_file", "edit"]
    assert get_adapter("swe-agent") is original


def test_unknown_operations_remain_visible_as_other(event_adapter: TraceAdapter) -> None:
    raw = _raw_trace(["OPEN", "UNKNOWN", "DONE"])
    assert event_adapter(raw) == ["read_file", "other", "submit"]


def test_missing_events_are_an_empty_sequence(event_adapter: TraceAdapter) -> None:
    assert event_adapter({}) == []


@pytest.mark.parametrize("events", [None, "invalid", {}])
def test_invalid_event_containers_raise(event_adapter: TraceAdapter, events: object) -> None:
    with pytest.raises(TypeError, match="expected list"):
        event_adapter({"events": events})


def test_malformed_events_follow_the_action_adapter_contract(event_adapter: TraceAdapter) -> None:
    raw = {"events": [None, "invalid", {}, {"op": "PATCH"}]}
    assert event_adapter(raw) == ["other", "edit"]
