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

Tab / inline autocomplete (next-edit prediction) is captured only COARSELY.
``cursorDiskKV`` has no per-completion Tab events (it persists Composer/chat
bubbles + Cmd-K ``inlineDiff`` edits), but the sibling ``ItemTable`` keeps
``aiCodeTracking.dailyStats.*`` with per-DAY ``tabSuggestedLines`` /
``tabAcceptedLines`` (and ``composer*`` equivalents). So Tab volume + accept-rate
are recoverable at day granularity, but not per-trace/per-event -- this adapter's
atom stream is the agentic/conversational surface, and weaving Tab into
trajectories would need an editor/telemetry hook. ``unifiedMode`` on
``composerData`` distinguishes the captured modes (``agent`` / ``chat`` / ``plan``).

Atom mapping (Cursor agent tool -> canonical atom):

    user turn (type 1)                          -> prompt_ai
    read_file                                   -> read_file
    grep / codebase_search / glob_file_search
      / list_dir / rg / web_search / web_fetch
      / semantic_search_full                    -> search_repo
    search_replace / write / apply_patch
      / edit_notebook / inline code block       -> edit
    delete_file                                 -> delete_file
    read_lints                                  -> lint
    run_terminal_cmd                            -> run_code
    todo_write / update_current_step / task
      / ask_question                            -> think
    ai reasoning turn, no tool                  -> think
    anything else (mcp_*, await, ...)           -> other

Design decisions:

- Tool name is the unit, not the bubble: one agent turn is one tool call here,
  so each event yields exactly one atom and the rule engine stays additive like
  the companion adapter.
- ``run_terminal_cmd`` maps to ``run_code`` rather than guessing test/vcs/lint:
  the command text is deliberately not extracted, so the finer Bash split would
  be a guess. Sessions that need it can route the raw command through the
  claude-code Bash classifier separately.
- A reasoning turn is ``think``, distinct from ``prompt_ai`` (the human handing
  off), mirroring the cursor-companion choice. Reasoning content lives in
  ``thinking`` on reasoning models and ``text`` otherwise, so both are checked:
  reading only ``text`` labels most modern assistant turns ``other``.

- The rule table and ``_atom_for_bubble`` must agree. The rules drive
  ``canonicalize`` (the shareable path) and ``_atom_for_bubble`` drives the
  local panel; a tool mapped in one and not the other would give the same
  session two different atom streams, so a test pins them together.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from procgrep.canonicalize import (
    EventRule,
    field_in,
    make_event_adapter,
    register_adapter,
)
from procgrep.types import (
    ATOM_DELETE_FILE,
    ATOM_EDIT,
    ATOM_LINT,
    ATOM_OTHER,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_SEARCH_REPO,
    ATOM_THINK,
)

_READ_TOOLS = {"read_file"}
_SEARCH_TOOLS = {
    "grep",
    "codebase_search",
    "glob_file_search",
    "list_dir",
    "rg",
    "web_search",
    "ripgrep_raw_search",
    "semantic_search_full",
    "web_fetch",
}
## Planning/bookkeeping tools: the agent orienting rather than acting on the
## repo, so they read as `think` alongside a plain reasoning turn.
_PLAN_TOOLS = {"todo_write", "update_current_step", "task", "ask_question"}
_EDIT_TOOLS = {"search_replace", "write", "apply_patch", "edit_notebook", "edit_file", "_codeblock"}
_TERMINAL_TOOLS = {"run_terminal_cmd", "run_terminal_command"}


def _normalize_tool(tool: str) -> str:
    """Strip Cursor's tool-version suffix so e.g. ``read_file_v2`` maps like
    ``read_file``. Cursor versions tool names (``_v2`` ...) across releases; the
    atom mapping is keyed on the base name, so normalize before lookup."""
    head, sep, tail = tool.rpartition("_v")
    if sep and head and tail.isdigit():
        return head
    return tool


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
    EventRule(field_in("tool", _PLAN_TOOLS), (ATOM_THINK,)),
    EventRule(field_in("tool", _TERMINAL_TOOLS), (ATOM_RUN_CODE,)),
)

cursor_vscdb_adapter = make_event_adapter(rules=CURSOR_VSCDB_RULES)
register_adapter("cursor-vscdb", cursor_vscdb_adapter, overwrite=True)


def _hash_id(value: str) -> str:
    """Stable short hash of a store-local identifier: path, or composer id.

    One helper for both so a session's trace id and its panel id are the same
    string and the two views can be joined. Stable across runs of the same
    store, meaningless outside it, which is what "hashed identifiers" buys:
    within-store joins without carrying the real path or session id.
    """
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _loads(value: object) -> dict[str, Any] | None:
    """Parse one ``cursorDiskKV`` value, or ``None`` when it is unusable.

    A live store holds the occasional row whose ``value`` is SQL NULL (an
    orphaned or half-written key). ``json.loads(None)`` raises ``TypeError``,
    not ``ValueError``, so a single such row would abort a whole read; one bad
    row should cost one session, not the run.
    """
    if not isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _params(tf: dict[str, Any]) -> dict[str, Any]:
    """A tool call's params, tolerating the JSON-string form Cursor sometimes uses."""
    p = tf.get("params")
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except ValueError:
            return {}
    return p if isinstance(p, dict) else {}


def _is_reasoning(bubble: dict[str, Any]) -> bool:
    """Whether an assistant bubble is a reasoning turn.

    Cursor puts a reasoning turn's content in ``thinking`` on reasoning models
    and in ``text`` otherwise, so checking only ``text`` labels most modern
    assistant turns ``other``: on a real store that was ~26k of 27.5k
    otherwise-empty bubbles, i.e. the bulk of the ``other`` share. Presence is
    all that is read; no reasoning content is extracted.
    """
    return bool(bubble.get("text") or bubble.get("thinking"))


def _event_for_bubble(bubble: dict[str, Any]) -> dict[str, Any]:
    """Map one bubble to a structure-only event: turn type, tool, hashed file.

    The hashed ``file`` (from ``targetFile`` / ``relativeWorkspacePath``) is the
    only addition beyond atom structure; it lets ``session_rework`` tell a
    re-edit of an already-touched file from forward progress, without carrying
    any path or content.
    """
    if bubble.get("type") == 1:
        return {"kind": "prompt"}
    tf = bubble.get("toolFormerData") or {}
    tool = tf.get("name") or tf.get("tool")
    if tool:
        tool = _normalize_tool(tool)
        event: dict[str, Any] = {"kind": "ai", "tool": tool}
        params = _params(tf)
        path = params.get("targetFile") or params.get("relativeWorkspacePath")
        if path:
            event["file"] = _hash_id(str(path))
        return event
    if bubble.get("codeBlocks"):
        return {"kind": "ai", "tool": "_codeblock"}
    if _is_reasoning(bubble):
        return {"kind": "think"}
    return {"kind": "ai"}  # no tool, no reasoning -> falls through to ATOM_OTHER


_FILE_EDIT_TOOLS = {"search_replace", "write", "apply_patch", "edit_notebook", "edit_file"}


def session_rework(events: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Model-free rework signal for one session's events.

    Rework is re-editing a file the session already edited: a proxy for
    correction rather than forward progress, and the negative term an
    interactive process reward wants. A prompt counts as a rework prompt when
    the agent re-touches an already-edited file before the next human prompt.
    File-less edits (inline code blocks) are not attributable and are skipped.

    Returns prompts, edits, rework_prompts, re_edits, and the two ratios
    (0.0 when the denominator is zero).
    """
    edited: set[str] = set()
    prompts = rework_prompts = edits = re_edits = 0
    flagged_this_run = False
    for event in events:
        if event.get("kind") == "prompt":
            prompts += 1
            flagged_this_run = False
            continue
        if event.get("tool") in _FILE_EDIT_TOOLS:
            file = event.get("file")
            if file is None:
                continue
            edits += 1
            if file in edited:
                re_edits += 1
                if not flagged_this_run:
                    rework_prompts += 1
                    flagged_this_run = True
            edited.add(file)
    return {
        "prompts": prompts,
        "edits": edits,
        "rework_prompts": rework_prompts,
        "re_edits": re_edits,
        "rework_ratio": rework_prompts / prompts if prompts else 0.0,
        "re_edit_ratio": re_edits / edits if edits else 0.0,
    }


def read_state_vscdb(path: str | Path, *, agent: str = "cursor") -> Iterator[dict[str, Any]]:
    """Yield one ``{trace_id, agent, events}`` record per composer session.

    ``trace_id`` is the hashed composer id, matching the panel's session id, so
    a record carries no real Cursor session identifier while still joining to
    the local view of the same store.

    Each session's turns are emitted in ``fullConversationHeadersOnly`` order so
    downstream atoms preserve the real human/agent interleaving.

    Bubbles are read one session at a time by indexed key range rather than all
    at once: a live store holds hundreds of thousands of bubbles, and loading
    them up front costs minutes and gigabytes before the first record is
    yielded. The ``>=``/``<`` range uses the key index, which ``LIKE`` does not
    (it is case-insensitive by default, so it full-scans).
    """
    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        # Streamed, not fetchall: `Connection.execute` makes a fresh cursor per
        # call, so the per-session bubble query below cannot invalidate this one,
        # and peak memory stays one session rather than every session's blob.
        for key, value in con.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key >= 'composerData:' AND key < 'composerData;'"
        ):
            session = _loads(value)
            if session is None or ":" not in key:
                continue
            composer_id = key.split(":", 1)[1]
            bubbles: dict[str, dict[str, Any]] = {}
            for bkey, bvalue in con.execute(
                "SELECT key, value FROM cursorDiskKV WHERE key >= ? AND key < ?",
                (f"bubbleId:{composer_id}:", f"bubbleId:{composer_id};"),
            ):
                bubble = _loads(bvalue)
                if bubble is not None and bkey.count(":") >= 2:
                    bubbles[bkey.split(":", 2)[2]] = bubble
            events = [
                _event_for_bubble(bubbles[bid])
                for header in session.get("fullConversationHeadersOnly", [])
                if (bid := str(header.get("bubbleId"))) in bubbles
            ]
            if events:
                yield {"trace_id": _hash_id(composer_id), "agent": agent, "events": events}
    finally:
        con.close()


_TOOL_ATOM = {
    **dict.fromkeys(_READ_TOOLS, ATOM_READ_FILE),
    **dict.fromkeys(_SEARCH_TOOLS, ATOM_SEARCH_REPO),
    **dict.fromkeys(_EDIT_TOOLS, ATOM_EDIT),
    "delete_file": ATOM_DELETE_FILE,
    "read_lints": ATOM_LINT,
    **dict.fromkeys(_PLAN_TOOLS, ATOM_THINK),
    **dict.fromkeys(_TERMINAL_TOOLS, ATOM_RUN_CODE),
}


def _atom_for_bubble(bubble: dict[str, Any]) -> str:
    """Atom for one assistant bubble, by its tool, code block, or reasoning."""
    tf = bubble.get("toolFormerData") or {}
    tool = tf.get("name") or tf.get("tool")
    if tool:
        tool = _normalize_tool(tool)
    if tool in _TOOL_ATOM:
        return _TOOL_ATOM[tool]
    if not tool and bubble.get("codeBlocks"):
        return ATOM_EDIT
    if not tool and _is_reasoning(bubble):
        return ATOM_THINK
    return ATOM_OTHER


def _dt_ms(ms: object) -> datetime | None:
    """A local ``datetime`` from epoch milliseconds, or ``None``."""
    if not isinstance(ms, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(ms / 1000)
    except (ValueError, OSError, OverflowError):
        return None


def build_panel_sessions(
    path: str | Path,
    *,
    paraphrase: Callable[[str], str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build live-panel sessions from a Cursor ``state.vscdb`` (LOCAL view).

    Mirrors the claude-code panel builder: each composer session becomes a set
    of prompt-anchored turns whose ``seq`` is the agent's atom sequence. Carries
    prompt text and the session title for the local panel only; pass
    ``paraphrase`` to normalize prompts, and use ``to_shareable`` for anything
    that leaves the machine (atoms and counts only).

    Memory-bounded for large stores: composer rows are read by indexed key
    prefix, the most recent ``limit`` are kept, and only those sessions' bubbles
    are loaded, one session at a time -- so cost scales with sessions shown, not
    DB size (the live store can be many GB).
    """
    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    sessions: list[dict[str, Any]] = []
    try:
        composers: list[tuple[str, dict[str, Any]]] = []
        # Range scan, not LIKE: LIKE ignores the key index (case-insensitive by
        # default), which full-scans a multi-GB store; a >=/< range uses it.
        for key, value in con.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key >= 'composerData:' AND key < 'composerData;'"
        ):
            parsed = _loads(value)
            if parsed is not None and ":" in key:
                composers.append((key.split(":", 1)[1], parsed))
        composers.sort(key=lambda c: c[1].get("lastUpdatedAt") or 0, reverse=True)
        if limit is not None:
            composers = composers[:limit]
        for cid, cd in composers:
            bubbles: dict[str, dict[str, Any]] = {}
            for key, value in con.execute(
                "SELECT key, value FROM cursorDiskKV WHERE key >= ? AND key < ?",
                (f"bubbleId:{cid}:", f"bubbleId:{cid};"),
            ):
                parsed = _loads(value)
                if parsed is not None and key.count(":") >= 2:
                    bubbles[key.split(":", 2)[2]] = parsed
            turns: list[dict[str, Any]] = []
            cur: dict[str, Any] | None = None
            for header in cd.get("fullConversationHeadersOnly", []):
                bubble = bubbles.get(str(header.get("bubbleId")))
                if bubble is None:
                    continue
                if bubble.get("type") == 1:
                    if cur is not None and cur["seq"]:
                        turns.append(cur)
                    dt = _dt_ms(bubble.get("createdAt"))
                    text = (bubble.get("text") or "").strip()
                    prompt = paraphrase(text) if (paraphrase is not None and text) else text
                    cur = {
                        "t": dt.strftime("%H:%M") if dt else "",
                        "model": "",
                        "prompt": prompt,
                        "plan": "",
                        "seq": [],
                    }
                elif cur is not None:
                    cur["seq"].append(_atom_for_bubble(bubble))
            if cur is not None and cur["seq"]:
                turns.append(cur)
            if not turns:
                continue
            created, updated = _dt_ms(cd.get("createdAt")), _dt_ms(cd.get("lastUpdatedAt"))
            sid = _hash_id(cid)
            sessions.append(
                {
                    "meta": {
                        "name": (cd.get("name") or "").strip() or sid,
                        "client": "Cursor",
                        "project": "Cursor",
                        "id": sid,
                        "date": updated.strftime("%b %d") if updated else "",
                        "ended": updated.isoformat() if updated else "",
                        "durationMin": round((updated - created).total_seconds() / 60)
                        if (created and updated and updated >= created)
                        else None,
                        "intent": "",
                        "illustrative": False,
                        "promptsParaphrased": paraphrase is not None,
                        "models": [],
                    },
                    "turns": turns,
                }
            )
    finally:
        con.close()
    return sessions


__all__ = [
    "CURSOR_VSCDB_RULES",
    "build_panel_sessions",
    "cursor_vscdb_adapter",
    "read_state_vscdb",
    "session_rework",
]
