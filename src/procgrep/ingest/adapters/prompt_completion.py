"""Prompt/completion chat-format adapter (SFT tool-use corpora).

Datasets like PrimeIntellect/INTELLECT-3-SFT (``toucan_tool``, ``swe_swiss``)
store a trajectory as two OpenAI-style message lists: ``prompt`` (the setup)
and ``completion`` (the rollout). Assistant turns carry ``tool_calls`` whose
``function.name`` is an arbitrary tool; ``tool`` turns hold results.

Each user turn emits PROMPT_AI, each assistant turn emits THINK for its text
plus one classified atom per tool call, and a tool result that reads as a
failure emits ERROR. Generic tool verbs map onto the nearest coding-alphabet
action type (get/list to read_file, search/query to search_repo, create to
create_file, ...), defaulting to OTHER: the structure is preserved exactly;
the semantics only as far as the shared alphabet allows (see the README
scope note).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import register_adapter
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_DELETE_FILE,
    ATOM_EDIT,
    ATOM_ERROR,
    ATOM_OTHER,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_THINK,
    Atom,
    AtomSequence,
)

# first matching verb wins; test outranks run so "run_tests" classifies as a test
_VERB_ATOMS: tuple[tuple[frozenset[str], Atom], ...] = (
    (frozenset({"test", "tests"}), ATOM_RUN_TEST),
    (frozenset({"search", "find", "query", "lookup", "grep"}), ATOM_SEARCH_REPO),
    (
        frozenset({"read", "get", "fetch", "list", "view", "retrieve", "show", "load"}),
        ATOM_READ_FILE,
    ),
    (frozenset({"create", "add", "new", "insert", "post", "make"}), ATOM_CREATE_FILE),
    (frozenset({"delete", "remove", "drop", "clear"}), ATOM_DELETE_FILE),
    (
        frozenset({"edit", "update", "write", "set", "modify", "patch", "put", "rename", "move"}),
        ATOM_EDIT,
    ),
    (frozenset({"run", "execute", "exec", "invoke", "call"}), ATOM_RUN_CODE),
)

_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_FAILURE_MARKS = ("error", "exception", "traceback", "failed", "not found")


def _tool_atom(name: str) -> Atom:
    words = {w.lower() for w in _WORD.findall(name)}
    for verbs, atom in _VERB_ATOMS:
        if words & verbs:
            return atom
    return ATOM_OTHER


def _as_turns(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    if not isinstance(value, list):
        return []
    return [m for m in value if isinstance(m, Mapping)]


def _call_name(call: Mapping[str, Any]) -> str:
    fn = call.get("function")
    if isinstance(fn, Mapping) and isinstance(fn.get("name"), str):
        return str(fn["name"])
    return str(call.get("name", ""))


def _is_failure(content: Any) -> bool:
    head = str(content)[:120].lower()
    return any(mark in head for mark in _FAILURE_MARKS)


def prompt_completion_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert a prompt/completion chat record into canonical atoms."""
    atoms: AtomSequence = []
    for turn in _as_turns(record.get("prompt")):
        if turn.get("role") == "user":
            atoms.append(ATOM_PROMPT_AI)
    for turn in _as_turns(record.get("completion")):
        role = turn.get("role")
        if role == "user":
            atoms.append(ATOM_PROMPT_AI)
        elif role == "assistant":
            content = turn.get("content")
            if isinstance(content, str) and content.strip():
                atoms.append(ATOM_THINK)
            for call in turn.get("tool_calls") or []:
                if isinstance(call, Mapping):
                    atoms.append(_tool_atom(_call_name(call)))
        elif role == "tool" and _is_failure(turn.get("content")):
            atoms.append(ATOM_ERROR)
    return atoms


def _register() -> None:
    register_adapter("prompt-completion", prompt_completion_adapter, overwrite=True)


_register()


__all__ = ["prompt_completion_adapter"]
