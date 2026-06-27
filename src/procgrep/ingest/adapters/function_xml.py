"""Function-XML text-trajectory adapter.

Some agents (R2E-Gym, several Qwen-Coder SWE-bench submissions) serialize tool
calls as XML-ish tags inside the assistant's text, rather than as structured
``tool_calls`` (openhands) or fenced ```bash blocks (react-text)::

    <function=execute_bash>
      <parameter=cmd>pytest tests/</parameter>
    </function>

``execute_bash`` routes through the shared command classifier; ``file_editor``
maps by its ``command`` parameter (view -> read_file, create -> create_file,
str_replace / insert -> edit); ``finish`` -> submit. Each assistant turn whose
reasoning text sits outside the function tags emits THINK. Degrades gracefully
on missing or malformed fields.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import register_adapter
from procgrep.ingest.adapters.mini_swe_agent import _classify_command
from procgrep.ingest.adapters.openhands import _EDITOR_COMMAND_ATOM
from procgrep.types import ATOM_EDIT, ATOM_OTHER, ATOM_SUBMIT, ATOM_THINK, Atom, AtomSequence

ASSISTANT_ROLES = frozenset({"assistant", "ai"})
_BASH_FUNCS = frozenset({"execute_bash", "run_bash", "bash", "execute_command"})
_EDITOR_FUNCS = frozenset(
    {"file_editor", "str_replace_editor", "str_replace_based_edit_tool", "editor", "edit_file"}
)
_SUBMIT_FUNCS = frozenset({"finish", "submit", "complete"})
_FUNCTION = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)


def _turn_text(turn: Mapping[str, Any]) -> str:
    for key in ("content", "text"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _params(body: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in _PARAMETER.finditer(body)}


def _classify_function(name: str, params: Mapping[str, str]) -> Atom:
    lowered = name.lower()
    if lowered in _BASH_FUNCS:
        return _classify_command(params.get("cmd") or params.get("command") or "")
    if lowered in _EDITOR_FUNCS:
        return _EDITOR_COMMAND_ATOM.get((params.get("command") or "").lower(), ATOM_EDIT)
    if lowered in _SUBMIT_FUNCS:
        return ATOM_SUBMIT
    if lowered == "think":
        return ATOM_THINK
    return ATOM_OTHER


def function_xml_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert a function-XML ``messages`` record into canonical atoms."""
    messages = record.get("messages") or []
    if not isinstance(messages, list):
        return []
    atoms: AtomSequence = []
    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") not in ASSISTANT_ROLES:
            continue
        text = _turn_text(msg)
        if not text:
            continue
        functions = _FUNCTION.findall(text)
        # Reasoning prose outside the function tags counts as one think step.
        if _FUNCTION.sub("", text).strip():
            atoms.append(ATOM_THINK)
        for name, body in functions:
            atoms.append(_classify_function(name, _params(body)))
    return atoms


def _register() -> None:
    register_adapter("function-xml", function_xml_adapter, overwrite=True)


_register()


__all__ = ["function_xml_adapter"]
