"""Gemini CLI session transcript adapter.

Gemini CLI is a separate terminal coding agent, woven in only as an ingest
adapter (an exemplar non-Claude-Code trace source), not part of procgrep's core.

The SWE-chat corpus carries Gemini CLI sessions in TWO shapes:

1. Native Gemini (the common case): a single JSON object per session::

     {"sessionId": "...", "messages": [
        {"type": "user", "content": [{"text": "..."}] | "..."},
        {"type": "gemini", "content": "...", "toolCalls": [{"name": "...", "args": {...}}]}]}

   A ``user`` message is the human prompt; a ``gemini`` message's ``toolCalls``
   list is the agent's tool calls for that turn.

2. Claude-Code-schema JSONL (a few sessions): byte-identical to a Claude Code
   transcript (``user`` / ``assistant`` / ``file-history-snapshot`` lines with
   ``tool_use`` content blocks). ``gemini_cli_adapter`` detects this and routes
   to the claude-code adapter, so both shapes map to the same atom alphabet.

``_flatten`` (native) explodes messages into atomic, prompt-anchored events,
which the shared feature-based `make_event_adapter` then maps.

Atom mapping (native Gemini tool -> canonical atom):

    user message (real human turn)               -> prompt_ai
    read_file                                     -> read_file
    grep_search / glob / list_directory / list_dir
      / find_file / google_web_search / web_fetch
      / search_for_pattern / find_symbol
      / get_symbols_overview                      -> search_repo
    replace / write_file / insert_after_symbol
      / insert_before_symbol / rename_symbol
      / replace_symbol_body                       -> edit
    run_shell_command, classified by leading verb -> run_test / version_control /
                                                     package / lint / search_repo /
                                                     run_code (else other)
    write_todos / enter_plan_mode / exit_plan_mode -> think
    ask_user / mcp_* / anything else               -> other

Design decisions:

- The first native ``user`` message is the Gemini CLI onboarding preamble ("You
  are an AI agent that brings the power of Gemini directly into the terminal..."),
  injected by the harness, not typed by the human. It is filtered out (the
  Gemini analogue of `_is_human_prompt`) so it is not miscounted as a prompt.

- ``run_shell_command`` reuses the claude-code Bash classifier so Gemini shares
  one command-category alphabet; the command is classified then discarded.

- ``mcp_serena_*`` symbol tools and bare ``mcp_*`` calls map by their action:
  serena find/overview to search, serena replace/insert to edit. Unrecognized
  MCP calls stay ``other`` rather than inflating a specific signal.

- Only the tool ``name`` and a shell command's leading verb cross the boundary;
  tool args and message text never reach the atom sequence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import EventRule, field_in, make_event_adapter, register_adapter
from procgrep.ingest.adapters.claude_code import _classify_terminal_command, claude_code_adapter
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

_READ_TOOLS = {
    "read_file",
    "mcp__filesystem__read_text_file",
    "read_memory",
    "mcp_serena_read_memory",
}
_SEARCH_TOOLS = {
    "grep_search",
    "glob",
    "list_directory",
    "list_dir",
    "find_file",
    "google_web_search",
    "web_fetch",
    "search_for_pattern",
    "find_symbol",
    "get_symbols_overview",
    "find_referencing_symbols",
    "mcp_serena_find_symbol",
    "mcp_serena_get_symbols_overview",
    "mcp_serena_find_referencing_symbols",
    "mcp_serena_find_file",
}
_EDIT_TOOLS = {
    "replace",
    "write_file",
    "insert_after_symbol",
    "insert_before_symbol",
    "rename_symbol",
    "replace_symbol_body",
    "mcp_serena_replace_symbol_body",
    "mcp_serena_insert_after_symbol",
    "mcp_serena_insert_before_symbol",
    "write_memory",
}
_THINK_TOOLS = {"write_todos", "enter_plan_mode", "exit_plan_mode"}
_SHELL_TOOLS = {"run_shell_command", "shell"}


GEMINI_RULES: tuple[EventRule, ...] = (
    EventRule(field_in("kind", {"user_prompt"}), (ATOM_PROMPT_AI,)),
    EventRule(field_in("tool", _READ_TOOLS), (ATOM_READ_FILE,)),
    EventRule(field_in("tool", _SEARCH_TOOLS), (ATOM_SEARCH_REPO,)),
    EventRule(field_in("tool", _EDIT_TOOLS), (ATOM_EDIT,)),
    EventRule(field_in("tool", _THINK_TOOLS), (ATOM_THINK,)),
    EventRule(field_in("kind", {"bash_read"}), (ATOM_READ_FILE,)),
    EventRule(field_in("kind", {"bash_test"}), (ATOM_RUN_TEST,)),
    EventRule(field_in("kind", {"bash_search"}), (ATOM_SEARCH_REPO,)),
    EventRule(field_in("kind", {"bash_vcs"}), (ATOM_VERSION_CONTROL,)),
    EventRule(field_in("kind", {"bash_package"}), (ATOM_PACKAGE,)),
    EventRule(field_in("kind", {"bash_lint"}), (ATOM_LINT,)),
    EventRule(field_in("kind", {"bash_run"}), (ATOM_RUN_CODE,)),
)

_MAP = make_event_adapter(rules=GEMINI_RULES)

# The harness-injected onboarding preamble that opens an auto-init Gemini session.
# It arrives as a user message but is not the human typing, so it must not become
# a prompt boundary (the Gemini analogue of claude-code's injected-turn guard).
_INJECTED_PREFIXES = ("You are an AI agent that brings the power of Gemini",)


def _user_text(content: Any) -> str:
    """The human's text from a native Gemini user message (str or [{text}] list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(b.get("text", "")) for b in content if isinstance(b, Mapping))
    return ""


def _is_human_prompt(content: Any) -> bool:
    """A native ``user`` turn typed by the human, not the injected onboarding preamble."""
    stripped = _user_text(content).strip()
    return bool(stripped) and not stripped.startswith(_INJECTED_PREFIXES)


def _is_native(record: Mapping[str, Any]) -> bool:
    """True when the events are native Gemini messages, not a Claude-Code schema.

    Both schemas use ``type == "user"``, so that alone does not distinguish them.
    The reliable tells: native uses a ``gemini`` message type and carries the
    turn under a top-level ``content`` key, whereas the Claude-Code schema uses
    ``assistant`` and nests the turn under ``message``. Inspect the first few
    events and decide on the first unambiguous signal.
    """
    for event in record.get("events", []):
        if not isinstance(event, Mapping):
            continue
        mtype = event.get("type")
        if mtype == "gemini":
            return True
        if mtype == "assistant":
            return False
        if mtype == "user":
            # Native carries top-level ``content``; the CC schema nests ``message``.
            return "message" not in event
    return False


def _flatten(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Explode native Gemini messages into atomic, prompt-anchored events.

    A real ``user`` message becomes one ``user_prompt`` event; each entry in a
    ``gemini`` message's ``toolCalls`` becomes one tool event, with
    ``run_shell_command`` classified by leading verb (command then discarded).
    """
    events: list[dict[str, Any]] = []
    messages = record.get("events")
    if not isinstance(messages, list):
        return events
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        mtype = message.get("type")
        if mtype == "user":
            if _is_human_prompt(message.get("content")):
                events.append({"kind": "user_prompt"})
        elif mtype == "gemini":
            for call in message.get("toolCalls") or []:
                if not isinstance(call, Mapping):
                    continue
                name = str(call.get("name") or "").lower()
                if name in _SHELL_TOOLS:
                    args = call.get("args")
                    command = args.get("command", "") if isinstance(args, Mapping) else ""
                    events.append({"kind": _classify_terminal_command(str(command))})
                else:
                    events.append({"kind": "tool", "tool": name})
    return events


def gemini_cli_adapter(record: Mapping[str, Any]) -> list[str]:
    """Convert one Gemini CLI session record into an atom sequence.

    Routes a Claude-Code-schema record to the claude-code adapter; otherwise
    maps the native Gemini message shape.
    """
    if _is_native(record):
        return _MAP({"events": _flatten(record)})
    return claude_code_adapter(record)


def load_gemini_session(obj: Any) -> dict[str, Any]:
    """Wrap a parsed Gemini CLI session into a ``{trace_id, agent, events}`` record.

    ``obj`` is either a parsed native session dict (with ``messages``) or a list
    of Claude-Code-schema JSONL lines. ``events`` holds whichever message list
    the adapter then flattens or routes.
    """
    if isinstance(obj, Mapping):
        sid = str(obj.get("sessionId") or "") or "gemini"
        return {"trace_id": sid, "agent": "gemini-cli", "events": obj.get("messages", [])}
    return {"trace_id": "gemini", "agent": "gemini-cli", "events": list(obj)}


register_adapter("gemini-cli", gemini_cli_adapter, overwrite=True)

__all__ = [
    "GEMINI_RULES",
    "gemini_cli_adapter",
    "load_gemini_session",
]
