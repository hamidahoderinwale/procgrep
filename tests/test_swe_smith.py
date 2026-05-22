"""Tests for the SWE-smith chat-format trajectory adapter."""

from __future__ import annotations

import json

from procgrep.adapters.swe_smith import swe_smith_adapter
from procgrep.types import (
    ATOM_EDIT,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
)


def test_returns_empty_for_missing_messages() -> None:
    assert swe_smith_adapter({}) == []


def test_returns_empty_for_malformed_json_string() -> None:
    assert swe_smith_adapter({"messages": "not-valid-json"}) == []


def test_returns_empty_when_messages_is_not_a_list() -> None:
    assert swe_smith_adapter({"messages": json.dumps({"foo": "bar"})}) == []


def test_returns_empty_when_messages_is_unexpected_type() -> None:
    assert swe_smith_adapter({"messages": 42}) == []


def test_handles_messages_passed_directly_as_list() -> None:
    messages = [{"role": "assistant", "action": "edit"}]
    assert swe_smith_adapter({"messages": messages}) == [ATOM_EDIT]


def test_extracts_action_from_json_string_messages() -> None:
    messages = [
        {"role": "system", "content": "..."},
        {"role": "assistant", "action": "edit"},
    ]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_EDIT]


def test_prepends_think_for_non_empty_thought() -> None:
    messages = [{"role": "assistant", "action": "edit", "thought": "I'll edit"}]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_THINK, ATOM_EDIT]


def test_skips_think_for_empty_thought() -> None:
    messages = [{"role": "assistant", "action": "edit", "thought": ""}]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_EDIT]


def test_skips_think_for_whitespace_only_thought() -> None:
    messages = [{"role": "assistant", "action": "edit", "thought": "   \n  "}]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_EDIT]


def test_falls_back_to_tool_calls_when_action_field_missing() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "edit", "arguments": "..."}, "id": "1"}],
        }
    ]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_EDIT]


def test_prefers_action_field_over_tool_calls() -> None:
    messages = [
        {
            "role": "assistant",
            "action": "edit",
            "tool_calls": [{"function": {"name": "search_dir", "arguments": "..."}, "id": "1"}],
        }
    ]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_EDIT]


def test_skips_non_assistant_roles() -> None:
    messages = [
        {"role": "user", "content": "..."},
        {"role": "system", "content": "..."},
        {"role": "tool", "content": "...", "tool_call_ids": ["1"]},
        {"role": "assistant", "action": "edit"},
    ]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_EDIT]


def test_unknown_action_falls_back_to_other() -> None:
    messages = [{"role": "assistant", "action": "unrecognized_action_name"}]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_OTHER]


def test_maps_swe_agent_action_synonyms() -> None:
    messages = [
        {"role": "assistant", "action": "str_replace"},
        {"role": "assistant", "action": "open"},
        {"role": "assistant", "action": "grep"},
        {"role": "assistant", "action": "pytest"},
        {"role": "assistant", "action": "submit"},
    ]
    record = {"messages": json.dumps(messages)}
    expected = [ATOM_EDIT, ATOM_READ_FILE, ATOM_SEARCH_REPO, ATOM_RUN_TEST, ATOM_SUBMIT]
    assert swe_smith_adapter(record) == expected


def test_ignores_non_mapping_entries() -> None:
    messages = [
        "not a message",
        None,
        {"role": "assistant", "action": "edit"},
        42,
    ]
    record = {"messages": json.dumps(messages)}
    assert swe_smith_adapter(record) == [ATOM_EDIT]


def test_ignores_malformed_tool_calls() -> None:
    """A non-list tool_calls field falls through to producing no atom."""
    messages = [
        {"role": "assistant", "tool_calls": "not a list"},
        {"role": "assistant", "tool_calls": []},
        {"role": "assistant", "tool_calls": [{"no_function_key": 1}]},
        {"role": "assistant", "tool_calls": [{"function": "not a dict"}]},
        {"role": "assistant", "tool_calls": [{"function": {"no_name": 1}}]},
    ]
    record = {"messages": json.dumps(messages)}
    # No action_name extractable for any of these → no atoms emitted.
    assert swe_smith_adapter(record) == []


def test_full_trajectory_roundtrip() -> None:
    """End-to-end test with a realistic SWE-smith-style trajectory record."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "Fix the bug in foo.py"},
        {
            "role": "assistant",
            "thought": "Let me look at foo.py first",
            "action": "open",
            "content": "I'll open foo.py to investigate.",
            "message_type": "action",
        },
        {"role": "tool", "content": "<contents of foo.py>"},
        {
            "role": "assistant",
            "thought": "I see the issue, let me fix it",
            "action": "str_replace",
            "content": "I'll fix the bug now.",
            "message_type": "action",
        },
        {"role": "tool", "content": "Edit successful"},
        {
            "role": "assistant",
            "action": "pytest",
            "message_type": "action",
        },
        {"role": "tool", "content": "All tests pass"},
        {
            "role": "assistant",
            "action": "submit",
            "message_type": "action",
        },
    ]
    record = {
        "messages": json.dumps(messages),
        "instance_id": "test__test.abc",
        "traj_id": "test_traj_001",
        "model": "claude-3-7-sonnet-20250219",
        "resolved": True,
    }
    expected = [
        ATOM_THINK,
        ATOM_READ_FILE,  # first assistant turn (open + thought)
        ATOM_THINK,
        ATOM_EDIT,  # second assistant turn (str_replace + thought)
        ATOM_RUN_TEST,  # third assistant turn (pytest, no thought)
        ATOM_SUBMIT,  # fourth assistant turn (submit, no thought)
    ]
    assert swe_smith_adapter(record) == expected


def test_registered_under_canonical_name() -> None:
    """The adapter registers itself under the name ``swe-smith``."""
    from procgrep.canonicalize import get_adapter, list_adapters

    assert "swe-smith" in list_adapters()
    assert get_adapter("swe-smith") is swe_smith_adapter
