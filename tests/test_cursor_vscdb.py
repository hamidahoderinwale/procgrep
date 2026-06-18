"""Cursor state.vscdb adapter: tool-name to atom mapping and turn order."""

from __future__ import annotations

from procgrep.canonicalize import canonicalize
from procgrep.ingest.adapters import cursor_vscdb  # noqa: F401  (registers adapter)


def _atoms(events):
    [trace] = canonicalize([{"trace_id": "t", "agent": "cursor", "events": events}], adapter="cursor-vscdb")
    return list(trace.atoms)


def test_tool_names_map_to_canonical_atoms():
    events = [
        {"kind": "prompt"},
        {"kind": "ai", "tool": "read_file"},
        {"kind": "ai", "tool": "codebase_search"},
        {"kind": "ai", "tool": "search_replace"},
        {"kind": "ai", "tool": "run_terminal_cmd"},
        {"kind": "ai", "tool": "read_lints"},
        {"kind": "ai", "tool": "delete_file"},
        {"kind": "think"},
    ]
    assert _atoms(events) == [
        "prompt_ai", "read_file", "search_repo", "edit",
        "run_code", "lint", "delete_file", "think",
    ]


def test_inline_code_block_is_edit_and_unknown_tool_is_other():
    assert _atoms([{"kind": "ai", "tool": "_codeblock"}]) == ["edit"]
    assert _atoms([{"kind": "ai", "tool": "todo_write"}]) == ["other"]
    assert _atoms([{"kind": "ai"}]) == ["other"]


def test_turn_order_preserved():
    events = [{"kind": "prompt"}, {"kind": "ai", "tool": "read_file"}, {"kind": "prompt"}]
    assert _atoms(events) == ["prompt_ai", "read_file", "prompt_ai"]
