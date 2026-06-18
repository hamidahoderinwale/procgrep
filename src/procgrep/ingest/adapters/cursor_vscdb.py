"""Cursor ``state.vscdb`` trace adapter.

Reads agent/composer sessions straight from a Cursor application-data dump
(``User/globalStorage/state.vscdb``), the richest local source of a developer's
own human+AI Cursor sessions. Unlike the cursor-companion service, this needs no
running exporter: it reads the SQLite store Cursor already keeps.

Layout (table ``cursorDiskKV``, key-prefixed JSON blobs):

    composerData:<id>   one session; ``fullConversationHeadersOnly`` is the
                        ordered turn list of {bubbleId, type} (1=user, 2=ai).
    bubbleId:<cid>:<bid>  one turn; ``toolFormerData.name`` is the agent tool
                        invoked (read_file, search_replace, grep, ...), or a
                        plain reasoning/chat turn when absent.

Only structure crosses the boundary: turn type and tool name, never prompt text,
tool args, or code. That keeps extraction inside procgrep's privacy model
(atoms + ids, no content), same as ``to_shareable``.

Atom mapping (Cursor agent tool -> canonical atom):

    user turn (type 1)                          -> prompt_ai
    read_file                                   -> read_file
    grep / codebase_search / glob_file_search
      / list_dir / rg / web_search              -> search_repo
    search_replace / write / apply_patch
      / edit_notebook / inline code block       -> edit
    delete_file                                 -> delete_file
    read_lints                                  -> lint
    run_terminal_cmd                            -> run_code
    ai reasoning turn, no tool                  -> think
    anything else (todo_write, mcp_*, ...)      -> other

Design decisions:

- Tool name is the unit, not the bubble: one agent turn is one tool call here,
  so each event yields exactly one atom and the rule engine stays additive like
  the companion adapter.
- ``run_terminal_cmd`` maps to ``run_code`` rather than guessing test/vcs/lint:
  the command text is deliberately not extracted, so the finer Bash split would
  be a guess. Sessions that need it can route the raw command through the
  claude-code Bash classifier separately.
- A reasoning turn (no tool, has text) is ``think``, distinct from ``prompt_ai``
  (the human handing off), mirroring the cursor-companion choice.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from procgrep.canonicalize import (
    EventRule,
    field_in,
    make_event_adapter,
    register_adapter,
)
from procgrep.ingest.adapters.cursor_companion import ATOM_PROMPT_AI
from procgrep.types import (
    ATOM_DELETE_FILE,
    ATOM_EDIT,
    ATOM_LINT,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_SEARCH_REPO,
    ATOM_THINK,
)

_READ_TOOLS = {"read_file"}
_SEARCH_TOOLS = {"grep", "codebase_search", "glob_file_search", "list_dir", "rg", "web_search"}
_EDIT_TOOLS = {"search_replace", "write", "apply_patch", "edit_notebook", "_codeblock"}
_TERMINAL_TOOLS = {"run_terminal_cmd"}

# Evaluated in order; each event matches exactly one rule (user turn, reasoning
# turn, or a single tool call), with the generic "other" fallback for the rest.
CURSOR_VSCDB_RULES: tuple[EventRule, ...] = (
    EventRule(field_in("kind", {"prompt"}), (ATOM_PROMPT_AI,)),
    EventRule(field_in("kind", {"think"}), (ATOM_THINK,)),
    EventRule(field_in("tool", _READ_TOOLS), (ATOM_READ_FILE,)),
    EventRule(field_in("tool", _SEARCH_TOOLS), (ATOM_SEARCH_REPO,)),
    EventRule(field_in("tool", _EDIT_TOOLS), (ATOM_EDIT,)),
    EventRule(field_in("tool", {"delete_file"}), (ATOM_DELETE_FILE,)),
    EventRule(field_in("tool", {"read_lints"}), (ATOM_LINT,)),
    EventRule(field_in("tool", _TERMINAL_TOOLS), (ATOM_RUN_CODE,)),
)

cursor_vscdb_adapter = make_event_adapter(rules=CURSOR_VSCDB_RULES)
register_adapter("cursor-vscdb", cursor_vscdb_adapter, overwrite=True)


def _event_for_bubble(bubble: dict[str, Any]) -> dict[str, Any]:
    """Map one bubble to a structure-only event (turn type and tool name)."""
    if bubble.get("type") == 1:
        return {"kind": "prompt"}
    tf = bubble.get("toolFormerData") or {}
    tool = tf.get("name") or tf.get("tool")
    if tool:
        return {"kind": "ai", "tool": tool}
    if bubble.get("codeBlocks"):
        return {"kind": "ai", "tool": "_codeblock"}
    if bubble.get("text"):
        return {"kind": "think"}
    return {"kind": "ai"}  # no tool, no text -> falls through to ATOM_OTHER


def read_state_vscdb(path: str | Path, *, agent: str = "cursor") -> Iterator[dict[str, Any]]:
    """Yield one ``{trace_id, agent, events}`` record per composer session.

    Each session's turns are emitted in ``fullConversationHeadersOnly`` order so
    downstream atoms preserve the real human/agent interleaving.
    """
    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        bubbles: dict[str, dict[str, Any]] = {}
        for key, value in con.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%'"
        ):
            try:
                bubbles[key.split(":", 1)[1]] = json.loads(value)
            except (ValueError, IndexError):
                continue
        for key, value in con.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"
        ):
            try:
                session = json.loads(value)
            except ValueError:
                continue
            composer_id = key.split(":", 1)[1]
            events = []
            for header in session.get("fullConversationHeadersOnly", []):
                bubble = bubbles.get(f"{composer_id}:{header.get('bubbleId')}")
                if bubble is not None:
                    events.append(_event_for_bubble(bubble))
            if events:
                yield {"trace_id": composer_id, "agent": agent, "events": events}
    finally:
        con.close()


__all__ = ["CURSOR_VSCDB_RULES", "cursor_vscdb_adapter", "read_state_vscdb"]
