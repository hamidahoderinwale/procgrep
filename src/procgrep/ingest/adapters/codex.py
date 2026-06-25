"""Codex (OpenAI) session transcript adapter.

Codex is a separate terminal coding agent, woven in only as an ingest adapter
(an exemplar non-Claude-Code trace source), not part of procgrep's core. Its
transcripts in the SWE-chat corpus are JSONL where each line is a tagged record::

    {"type": "session_meta", "payload": {...}}
    {"type": "event_msg", "payload": {"type": "user_message", "message": "..."}}
    {"type": "response_item", "payload": {"type": "function_call", "name": "...",
                                          "arguments": "{...json...}"}}
    {"type": "response_item", "payload": {"type": "custom_tool_call",
                                          "name": "apply_patch", "input": "..."}}

Why Codex was "0 computable" before: it shares none of the claude-code shape.
There are no ``tool_use`` content blocks and no ``tool`` field; tool calls are
``response_item`` payloads typed ``function_call`` / ``custom_tool_call`` with a
top-level ``name``, and shell calls hide the command inside a JSON-encoded
``arguments`` string (key ``cmd`` for ``exec_command``, ``command`` for
``shell_command``). A claude-code-style flatten finds nothing, so every session
collapsed to empty. This adapter reads the Codex event types directly.

Human prompts are taken from ``event_msg`` ``user_message`` records, NOT from
``response_item`` role=user messages: the latter also carry harness-injected
turns (the AGENTS.md preamble, ``<skill>`` expansions, the developer
permissions block), which would be miscounted as human prompts. ``user_message``
is the actual typed turn, the Codex analogue of `_is_human_prompt`.

Atom mapping:

    event_msg user_message                          -> prompt_ai
    function_call exec_command / shell_command,      -> read_file / run_test /
      classified by the command's leading verb         version_control / package /
                                                        lint / search_repo /
                                                        run_code (else other)
    function_call update_plan                        -> think
    function_call write_stdin                        -> run_code
    custom_tool_call apply_patch                     -> edit
    mcp__filesystem__read_text_file / get_file_info  -> read_file
    mcp__filesystem__search_files / *ast_grep_search
      / *list_*                                      -> search_repo
    mcp__filesystem__write_file                      -> edit
    mcp__*sequentialthinking / *project_memory_read  -> think
    function_call mcp__* (other) / anything else      -> other

Design decisions:

- Shell calls reuse the claude-code terminal classifier on the extracted command,
  so Codex shares the same command-category alphabet; the command is classified
  and then discarded, never retained -- the same privacy boundary as the other
  adapters. Args that fail to JSON-parse degrade to an unclassified shell call
  (``other``) rather than crashing.

- Codex reads files by shelling out (``sed -n``, ``cat``, ``nl``) rather than
  via a Read tool, so the terminal classifier's read class is what recovers them
  as ``read_file``; without it nearly half of Codex's actions collapse to
  ``other``.

- Codex leans on MCP servers (``filesystem``, ``code_intel``, ``omx_*``) for
  read/search/edit. The common MCP tool names are mapped by their action so they
  do not all fall to ``other``; an unrecognized ``mcp__*`` stays ``other`` rather
  than inflating a specific signal.

- Only the leading verb of a shell command and the tool ``name`` cross the
  boundary; arguments and message text never reach the atom sequence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import EventRule, field_in, make_event_adapter, register_adapter
from procgrep.ingest.adapters.claude_code import _classify_terminal_command
from procgrep.types import (
    ATOM_EDIT,
    ATOM_LINT,
    ATOM_PACKAGE,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_THINK,
    ATOM_VERSION_CONTROL,
)

# Codex function_call names that run a shell command. The command lives in the
# JSON-encoded arguments under these keys, classified then discarded.
_SHELL_CALLS = {"exec_command", "shell_command", "shell", "local_shell"}
_EDIT_CALLS = {"apply_patch"}
_THINK_CALLS = {"update_plan"}
_RUN_CALLS = {"write_stdin"}


def _classify_mcp(name: str) -> str:
    """Map an ``mcp__*`` tool name to a tool category by its action substring.

    MCP server tool names are namespaced (``mcp__filesystem__read_text_file``)
    and vary by server, so match by the action verb in the name rather than an
    exact set. Anything unrecognized returns ``mcp`` (which falls through to
    ``other``), so an unknown MCP tool never inflates a specific signal.
    """
    n = name.lower()
    if "read" in n or "get_file_info" in n or "hover" in n or "definition" in n:
        return "mcp_read"
    if "write_file" in n or "replace" in n or "insert" in n or "create_file" in n:
        return "mcp_edit"
    if (
        "search" in n or "find" in n or "list" in n or "symbol" in n
        or "diagnostic" in n or "grep" in n
    ):
        return "mcp_search"
    if "thinking" in n or "memory_read" in n or "state_" in n:
        return "mcp_think"
    return "mcp"


CODEX_RULES: tuple[EventRule, ...] = (
    EventRule(field_in("kind", {"user_prompt"}), (ATOM_PROMPT_AI,)),
    EventRule(field_in("call", _EDIT_CALLS), (ATOM_EDIT,)),
    EventRule(field_in("call", _THINK_CALLS), (ATOM_THINK,)),
    EventRule(field_in("call", _RUN_CALLS), (ATOM_RUN_CODE,)),
    EventRule(field_in("kind", {"bash_read", "mcp_read"}), (ATOM_READ_FILE,)),
    EventRule(field_in("kind", {"bash_test"}), (ATOM_RUN_TEST,)),
    EventRule(field_in("kind", {"bash_search", "mcp_search"}), (ATOM_SEARCH_REPO,)),
    EventRule(field_in("kind", {"bash_vcs"}), (ATOM_VERSION_CONTROL,)),
    EventRule(field_in("kind", {"bash_package"}), (ATOM_PACKAGE,)),
    EventRule(field_in("kind", {"bash_lint"}), (ATOM_LINT,)),
    EventRule(field_in("kind", {"bash_run"}), (ATOM_RUN_CODE,)),
    EventRule(field_in("kind", {"mcp_edit"}), (ATOM_EDIT,)),
    EventRule(field_in("kind", {"mcp_think"}), (ATOM_THINK,)),
)

_MAP = make_event_adapter(rules=CODEX_RULES)


def _shell_command(arguments: Any) -> str:
    """The command string from a shell call's JSON-encoded arguments, or ``''``."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return ""
    if isinstance(arguments, Mapping):
        return str(arguments.get("cmd") or arguments.get("command") or "")
    return ""


def _flatten(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Explode Codex JSONL records into atomic, prompt-anchored events.

    ``event_msg`` ``user_message`` records become ``user_prompt`` events;
    ``response_item`` ``function_call`` / ``custom_tool_call`` records become one
    tool event each, with shell calls classified by leading verb and the command
    then discarded.
    """
    events: list[dict[str, Any]] = []
    lines = record.get("events")
    if not isinstance(lines, list):
        return events
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        outer = line.get("type")
        payload = line.get("payload")
        if not isinstance(payload, Mapping):
            continue
        ptype = payload.get("type")
        if outer == "event_msg" and ptype == "user_message":
            if str(payload.get("message") or "").strip():
                events.append({"kind": "user_prompt"})
        elif outer == "response_item" and ptype in ("function_call", "custom_tool_call"):
            name = str(payload.get("name") or "").lower()
            if name in _SHELL_CALLS:
                command = _shell_command(payload.get("arguments"))
                events.append({"kind": _classify_terminal_command(command)})
            elif name.startswith("mcp_"):
                events.append({"kind": _classify_mcp(name)})
            else:
                events.append({"kind": "tool", "call": name})
    return events


def codex_adapter(record: Mapping[str, Any]) -> list[str]:
    """Convert one Codex session record into an atom sequence."""
    return _MAP({"events": _flatten(record)})


def load_codex_session(lines: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Wrap parsed Codex JSONL lines as a ``{trace_id, agent, events}`` record.

    The session id is read from the ``session_meta`` payload and used as the
    trace id; the agent is ``codex``.
    """
    sid = "codex"
    for line in lines:
        if isinstance(line, Mapping) and line.get("type") == "session_meta":
            payload = line.get("payload")
            if isinstance(payload, Mapping) and payload.get("id"):
                sid = str(payload["id"])
            break
    return {"trace_id": sid, "agent": "codex", "events": list(lines)}


register_adapter("codex", codex_adapter, overwrite=True)

__all__ = [
    "CODEX_RULES",
    "codex_adapter",
    "load_codex_session",
]
