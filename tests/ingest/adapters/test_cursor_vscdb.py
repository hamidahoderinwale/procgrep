"""Cursor state.vscdb adapter: tool-name to atom mapping and turn order."""

from __future__ import annotations

from procgrep.canonicalize import canonicalize
from procgrep.ingest.adapters import cursor_vscdb  # noqa: F401  (registers adapter)
from procgrep.ingest.adapters.cursor_vscdb import session_rework


def _atoms(events):
    [trace] = canonicalize(
        [{"trace_id": "t", "agent": "cursor", "events": events}], adapter="cursor-vscdb"
    )
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
        "prompt_ai",
        "read_file",
        "search_repo",
        "edit",
        "run_code",
        "lint",
        "delete_file",
        "think",
    ]


def test_inline_code_block_is_edit_and_unknown_tool_is_other():
    assert _atoms([{"kind": "ai", "tool": "_codeblock"}]) == ["edit"]
    assert _atoms([{"kind": "ai", "tool": "mcp_linear_get_issue"}]) == ["other"]
    assert _atoms([{"kind": "ai"}]) == ["other"]


def test_planning_tools_are_think():
    """Bookkeeping is the agent orienting, not acting on the repo."""
    assert _atoms([{"kind": "ai", "tool": "todo_write"}]) == ["think"]
    assert _atoms([{"kind": "ai", "tool": "update_current_step"}]) == ["think"]


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


def _make_db(tmp_path):
    import json
    import sqlite3

    db = tmp_path / "state.vscdb"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    cid = "abc"
    con.execute(
        "INSERT INTO cursorDiskKV VALUES(?,?)",
        (
            f"composerData:{cid}",
            json.dumps(
                {
                    "name": "Demo session",
                    "createdAt": 1765051555277,
                    "lastUpdatedAt": 1765051621393,
                    "fullConversationHeadersOnly": [
                        {"bubbleId": "b1", "type": 1},
                        {"bubbleId": "b2", "type": 2},
                        {"bubbleId": "b3", "type": 2},
                    ],
                }
            ),
        ),
    )
    for bid, b in [
        ("b1", {"type": 1, "text": "do the thing", "createdAt": 1765051555277}),
        ("b2", {"type": 2, "toolFormerData": {"name": "read_file"}}),
        ("b3", {"type": 2, "toolFormerData": {"name": "search_replace"}}),
    ]:
        con.execute(
            "INSERT INTO cursorDiskKV VALUES(?,?)", (f"bubbleId:{cid}:{bid}", json.dumps(b))
        )
    con.commit()
    con.close()
    return db


def test_build_panel_sessions(tmp_path):
    from procgrep.ingest.adapters.cursor_vscdb import build_panel_sessions

    sessions = build_panel_sessions(_make_db(tmp_path))
    assert len(sessions) == 1
    s = sessions[0]
    assert s["meta"]["client"] == "Cursor"
    assert s["meta"]["name"] == "Demo session"
    assert s["meta"]["durationMin"] == 1
    assert s["turns"][0]["seq"] == ["read_file", "edit"]
    assert s["turns"][0]["prompt"] == "do the thing"


def test_build_panel_sessions_limit_zero(tmp_path):
    from procgrep.ingest.adapters.cursor_vscdb import build_panel_sessions

    assert build_panel_sessions(_make_db(tmp_path), limit=0) == []


def test_read_state_vscdb(tmp_path):
    from procgrep.ingest.adapters.cursor_vscdb import read_state_vscdb

    records = list(read_state_vscdb(_make_db(tmp_path)))
    assert len(records) == 1
    assert [e.get("kind") for e in records[0]["events"]] == ["prompt", "ai", "ai"]


def test_trace_id_is_hashed_and_joins_to_the_panel_id(tmp_path):
    """No real Cursor session id leaves, and the two views stay joinable."""
    from procgrep.ingest.adapters.cursor_vscdb import build_panel_sessions, read_state_vscdb

    db = _make_db(tmp_path)
    (trace,) = read_state_vscdb(db)
    (panel,) = build_panel_sessions(db)
    assert trace["trace_id"] != "abc"
    assert trace["trace_id"] == panel["meta"]["id"]


def test_reasoning_turn_is_think_from_either_field():
    """Reasoning models put the turn in `thinking`, older ones in `text`."""
    from procgrep.ingest.adapters.cursor_vscdb import _atom_for_bubble, _event_for_bubble

    for bubble in ({"type": 2, "text": "hm"}, {"type": 2, "thinking": {"text": "hm"}}):
        assert _event_for_bubble(bubble) == {"kind": "think"}
        assert _atom_for_bubble(bubble) == "think"
    assert _atom_for_bubble({"type": 2}) == "other"


def test_rule_table_and_panel_mapping_agree():
    """Both paths must give a tool the same atom, or one session gets two streams."""
    from procgrep.canonicalize import get_adapter
    from procgrep.ingest.adapters.cursor_vscdb import _TOOL_ATOM, _atom_for_bubble

    adapter = get_adapter("cursor-vscdb")
    for tool, atom in _TOOL_ATOM.items():
        if tool == "_codeblock":  # panel-only pseudo-tool, no bubble carries it
            continue
        assert adapter({"events": [{"kind": "ai", "tool": tool}]}) == [atom], tool
        assert _atom_for_bubble({"type": 2, "toolFormerData": {"name": tool}}) == atom, tool


def _add_junk_rows(db):
    """A NULL value and a malformed blob, as a live store accumulates."""
    import sqlite3

    con = sqlite3.connect(db)
    con.execute("INSERT INTO cursorDiskKV VALUES('composerData:orphan', NULL)")
    con.execute("INSERT INTO cursorDiskKV VALUES('composerData:broken', '{not json')")
    con.execute("INSERT INTO cursorDiskKV VALUES('bubbleId:abc:b9', NULL)")
    con.commit()
    con.close()


def test_null_and_malformed_rows_cost_one_session_not_the_run(tmp_path):
    """A single NULL-valued row must not abort the read of a whole store.

    `json.loads(None)` raises TypeError, not ValueError, so an unguarded read
    dies on the first orphaned key. Real stores carry a few.
    """
    from procgrep.ingest.adapters.cursor_vscdb import build_panel_sessions, read_state_vscdb

    db = _make_db(tmp_path)
    _add_junk_rows(db)
    assert len(list(read_state_vscdb(db))) == 1
    assert [s["meta"]["name"] for s in build_panel_sessions(db)] == ["Demo session"]
