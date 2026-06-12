"""OpenHands trajectory adapter.

OpenHands (https://github.com/All-Hands-AI/OpenHands) is the dominant scaffold
behind the large public trajectory datasets (nebius, nvidia SWE-Hero/SWE-Zero,
SWE-Gym). Each run is a ``messages`` list; assistant turns carry ``tool_calls``,
each a function call::

    {"messages": [
        {"role": "assistant",
         "tool_calls": [{"function": {"name": "execute_bash",
                                      "arguments": "{\\"command\\": \\"pytest tests/\\"}"}}]},
        ...]}

Tool -> canonical atom:
  - ``execute_bash`` / ``run_bash`` / ``bash``: classify the bash command
    (pytest -> run_test, grep/find -> search_repo, cat -> read_file, ...).
  - ``str_replace_editor`` / ``str_replace_based_edit_tool`` / ``edit_file``:
    ``command=view`` -> read_file, ``create`` -> create_file,
    ``str_replace``/``insert`` -> edit.
  - ``finish`` / ``submit`` -> submit; ``think`` -> think; else -> other.

Assistant turns with non-empty textual content emit ``THINK`` before their
tool calls. Degrades gracefully on missing or malformed fields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import register_adapter

# Reuse the well-tested bash-command classifier from the mini-swe-agent adapter
# rather than duplicate the command->atom table.
from procgrep.ingest.adapters.mini_swe_agent import _classify_command as _classify_bash
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_EDIT,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_SUBMIT,
    ATOM_THINK,
    Atom,
    AtomSequence,
)

_BASH_TOOLS = frozenset({"execute_bash", "run_bash", "bash", "execute_command"})
_EDITOR_TOOLS = frozenset(
    {"str_replace_editor", "str_replace_based_edit_tool", "edit_file", "editor", "edit"}
)
_EDITOR_COMMAND_ATOM: dict[str, Atom] = {
    "view": ATOM_READ_FILE,
    "open": ATOM_READ_FILE,
    "create": ATOM_CREATE_FILE,
    "write": ATOM_CREATE_FILE,
    "str_replace": ATOM_EDIT,
    "insert": ATOM_EDIT,
    "append": ATOM_EDIT,
}
_SUBMIT_TOOLS = frozenset({"finish", "submit", "complete"})


def _arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    """Parse a tool call's ``function.arguments`` (JSON string or dict)."""
    fn = call.get("function")
    raw = fn.get("arguments") if isinstance(fn, Mapping) else call.get("arguments")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_name(call: Mapping[str, Any]) -> str:
    """Extract the function/tool name from a tool call."""
    fn = call.get("function")
    if isinstance(fn, Mapping) and fn.get("name"):
        return str(fn["name"])
    return str(call.get("name", ""))


def _classify_tool_call(call: Mapping[str, Any]) -> Atom:
    """Map one OpenHands tool call to a canonical atom."""
    name = _tool_name(call).lower()
    args = _arguments(call)
    if name in _BASH_TOOLS:
        return _classify_bash(str(args.get("command", "")))
    if name in _EDITOR_TOOLS:
        command = str(args.get("command", "")).lower()
        return _EDITOR_COMMAND_ATOM.get(command, ATOM_EDIT)
    if name in _SUBMIT_TOOLS:
        return ATOM_SUBMIT
    if name == "think":
        return ATOM_THINK
    return ATOM_OTHER


def openhands_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert an OpenHands ``messages`` record into canonical atoms."""
    messages = record.get("messages") or []
    if not isinstance(messages, list):
        return []

    atoms: AtomSequence = []
    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            atoms.append(ATOM_THINK)
        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if isinstance(call, Mapping):
                atoms.append(_classify_tool_call(call))
    return atoms


def _register() -> None:
    """Register under the name ``"openhands"``."""
    register_adapter("openhands", openhands_adapter, overwrite=True)


_register()


__all__ = ["openhands_adapter"]
