"""Cursor state.vscdb adapter: tool-name to atom mapping and turn order."""

from __future__ import annotations

from procgrep.canonicalize import canonicalize
from procgrep.ingest.adapters import cursor_vscdb  # noqa: F401  (registers adapter)
from procgrep.ingest.adapters.cursor_vscdb import session_rework


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


def test_rework_counts_re_edits_of_same_file():
    events = [
        {"kind": "prompt"},
        {"kind": "ai", "tool": "write", "file": "a"},
        {"kind": "ai", "tool": "search_replace", "file": "b"},
        {"kind": "prompt"},
        {"kind": "ai", "tool": "search_replace", "file": "a"},  # re-edit of a -> rework
    ]
    r = session_rework(events)
    assert r["prompts"] == 2
    assert r["edits"] == 3
    assert r["re_edits"] == 1
    assert r["rework_prompts"] == 1
    assert r["rework_ratio"] == 0.5


def test_rework_zero_when_no_repeats_and_fileless_edits_skipped():
    assert session_rework([{"kind": "ai", "tool": "write", "file": "a"}])["re_edit_ratio"] == 0.0
    # inline code block edits have no file and are not counted
    assert session_rework([{"kind": "ai", "tool": "_codeblock"}])["edits"] == 0
