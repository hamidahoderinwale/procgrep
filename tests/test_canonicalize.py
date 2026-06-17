"""Tests for `procgrep.canonicalize`."""

from __future__ import annotations

import pytest

from procgrep.canonicalize import (
    EventRule,
    canonicalize,
    field_in,
    field_truthy,
    get_adapter,
    list_adapters,
    make_action_adapter,
    make_event_adapter,
    register_adapter,
)
from procgrep.types import ATOM_EDIT, ATOM_OTHER, ATOM_RUN_TEST, ATOM_THINK


def test_builtin_adapters_are_registered() -> None:
    names = set(list_adapters())
    assert {"swe-agent", "agentless", "dars", "moatless"}.issubset(names)


def test_get_adapter_raises_for_unknown_name() -> None:
    with pytest.raises(KeyError, match="no adapter named"):
        get_adapter("not-a-real-adapter")


def test_register_adapter_refuses_overwrite_by_default() -> None:
    adapter = make_action_adapter(action_field="action", atom_map={})
    register_adapter("tmp-adapter", adapter)
    with pytest.raises(ValueError, match="already registered"):
        register_adapter("tmp-adapter", adapter)
    register_adapter("tmp-adapter", adapter, overwrite=True)


def test_make_action_adapter_maps_known_and_unknown_names() -> None:
    adapter = make_action_adapter(
        action_field="action",
        atom_map={"do_edit": ATOM_EDIT, "do_test": ATOM_RUN_TEST},
    )
    record = {
        "trace_id": "t1",
        "agent": "alpha",
        "actions": [
            {"action": "do_edit"},
            {"action": "do_test"},
            {"action": "mystery_op"},
        ],
    }
    assert adapter(record) == [ATOM_EDIT, ATOM_RUN_TEST, ATOM_OTHER]


def test_make_action_adapter_emits_think_for_nonempty_thought() -> None:
    adapter = make_action_adapter(
        action_field="action",
        atom_map={"e": ATOM_EDIT},
        thought_field="thought",
    )
    record = {
        "actions": [
            {"action": "e", "thought": "I will edit"},
            {"action": "e", "thought": "   "},
        ],
    }
    assert adapter(record) == [ATOM_THINK, ATOM_EDIT, ATOM_EDIT]


def test_canonicalize_end_to_end() -> None:
    raw = [
        {
            "trace_id": "t1",
            "agent": "alpha",
            "group": "cell-A",
            "actions": [{"action": "e"}, {"action": "t"}],
        },
        {
            "trace_id": "t2",
            "agent": "beta",
            "actions": [{"action": "e"}],
        },
    ]
    adapter = make_action_adapter(
        action_field="action",
        atom_map={"e": ATOM_EDIT, "t": ATOM_RUN_TEST},
    )
    traces = canonicalize(raw, adapter=adapter)
    assert [t.trace_id for t in traces] == ["t1", "t2"]
    assert traces[0].group == "cell-A"
    assert traces[1].group is None
    assert traces[0].atoms == [ATOM_EDIT, ATOM_RUN_TEST]


# --- make_event_adapter (feature-based, multi-atom) -------------------------


def test_make_event_adapter_decomposes_dedupes_and_falls_back() -> None:
    rules = [
        EventRule(field_in("kind", {"ask"}), ("prompt_ai",)),
        EventRule(field_truthy("changed"), (ATOM_EDIT,)),
        EventRule(field_truthy("changed"), (ATOM_EDIT,)),  # duplicate -> collapsed
    ]
    adapter = make_event_adapter(rules=rules, events_path="steps", default_atom=ATOM_OTHER)
    record = {
        "steps": [
            {"kind": "ask", "changed": True},  # prompt_ai + edit (deduped)
            {"kind": "noop"},  # no rule -> default
        ]
    }
    assert adapter(record) == ["prompt_ai", ATOM_EDIT, ATOM_OTHER]


def test_make_event_adapter_is_lenient_on_malformed_records() -> None:
    adapter = make_event_adapter(rules=[])
    assert adapter({"events": "not a list"}) == []
    assert adapter({}) == []
    assert adapter({"events": None}) == []


def test_make_event_adapter_normalize_hook_flattens_before_rules() -> None:
    rules = [EventRule(field_in("type", {"edit"}), (ATOM_EDIT,))]

    def lift_inner(event: dict) -> dict:  # type: ignore[type-arg]
        return {**event.get("inner", {}), **{k: v for k, v in event.items() if k != "inner"}}

    adapter = make_event_adapter(rules=rules, normalize=lift_inner)  # type: ignore[arg-type]
    assert adapter({"events": [{"inner": {"type": "edit"}}]}) == [ATOM_EDIT]
