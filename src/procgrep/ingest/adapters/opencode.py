"""OpenCode session transcript adapter.

OpenCode is a separate terminal coding agent, woven in only as an ingest
adapter (an exemplar non-Claude-Code trace source), not part of procgrep's core.
Its transcripts in the SWE-chat corpus are a single JSON object per session::

    {"info": {"id", "directory", "title", ...},
     "messages": [{"info": {"role": "user"|"assistant", ...}, "parts": [...]}]}

A message's ``parts`` carry the structure: a ``text`` part on a user message is
the human prompt; a ``tool`` part on an assistant message is one tool call
(``part["tool"]`` is the tool name, ``part["state"]["input"]`` its arguments).

``_flatten`` explodes messages into atomic, prompt-anchored events (one human
prompt or one assistant tool call each), which the shared feature-based
`make_event_adapter` then maps -- the same machinery the claude-code adapter
uses on a structurally different source.

Atom mapping (OpenCode tool -> canonical atom):

    user text part                              -> prompt_ai
    read                                        -> read_file
    grep / glob / webfetch / tavily-search      -> search_repo
    edit / write / apply_patch                  -> edit
    bash, classified by its leading verb        -> read_file / run_test /
                                                   version_control / package /
                                                   lint / search_repo / run_code
                                                   (else other)
    todowrite                                    -> think
    task / question (and anything else)          -> other

Design decisions:

- Only action structure is read: the tool name and, for bash, the command's
  leading verb (classified then discarded, never retained). Message text and
  tool arguments never reach the atom sequence, so a session is fingerprinted
  without exposing content -- the same privacy boundary as the other adapters.

- ``bash`` reuses the claude-code Bash classifier so the two sources share one
  command-category alphabet rather than forking a second classifier.

- ``prompt_ai`` marks the human-to-AI handoff, the shared interactive atom; the
  injected first prompt OpenCode's own "init" flow emits is not special-cased
  here because, unlike Codex/Gemini, the corpus sessions begin with a real
  human turn.
"""

from __future__ import annotations

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

_READ_TOOLS = {"read"}
_SEARCH_TOOLS = {"grep", "glob", "webfetch", "tavily_tavily-search", "web_search", "list"}
_EDIT_TOOLS = {"edit", "write", "apply_patch"}
_THINK_TOOLS = {"todowrite"}


OPENCODE_RULES: tuple[EventRule, ...] = (
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

_MAP = make_event_adapter(rules=OPENCODE_RULES)


def _part_command(part: Mapping[str, Any]) -> str:
    """The bash command string from a tool part's input, or ``''``."""
    state = part.get("state")
    inp = state.get("input") if isinstance(state, Mapping) else None
    if isinstance(inp, Mapping):
        return str(inp.get("command") or "")
    return ""


def _flatten(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Explode OpenCode messages into atomic, prompt-anchored events.

    A user message's ``text`` part becomes one ``user_prompt`` event; each
    assistant ``tool`` part becomes one tool event, with ``bash`` further
    classified by its leading verb (the command is then discarded).
    """
    events: list[dict[str, Any]] = []
    messages = record.get("events")
    if not isinstance(messages, list):
        return events
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        info = message.get("info")
        role = info.get("role") if isinstance(info, Mapping) else None
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            ptype = part.get("type")
            if role == "user" and ptype == "text":
                if str(part.get("text") or "").strip():
                    events.append({"kind": "user_prompt"})
            elif role == "assistant" and ptype == "tool":
                tool = str(part.get("tool") or "").lower()
                if tool == "bash":
                    events.append({"kind": _classify_terminal_command(_part_command(part))})
                else:
                    events.append({"kind": "tool", "tool": tool})
    return events


def opencode_adapter(record: Mapping[str, Any]) -> list[str]:
    """Convert one OpenCode session record into an atom sequence."""
    return _MAP({"events": _flatten(record)})


def load_opencode_session(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a parsed OpenCode session JSON as a ``{trace_id, agent, events}`` record.

    ``events`` holds the raw ``messages`` list; the adapter flattens it. The
    session id is taken from ``info.id`` and used as both trace id and agent.
    """
    info = obj.get("info") if isinstance(obj, Mapping) else None
    sid = str(info.get("id") if isinstance(info, Mapping) else "") or "opencode"
    return {"trace_id": sid, "agent": "opencode", "events": obj.get("messages", [])}


register_adapter("opencode", opencode_adapter, overwrite=True)

__all__ = [
    "OPENCODE_RULES",
    "load_opencode_session",
    "opencode_adapter",
]
