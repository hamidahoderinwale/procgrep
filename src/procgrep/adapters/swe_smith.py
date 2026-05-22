"""SWE-smith chat-format trajectory adapter.

SWE-smith-trajectories (https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)
stores each rollout as a ``messages`` JSON-string list in chat format,
with assistant turns carrying ``action``, ``thought``, and ``tool_calls``
fields. This adapter parses the ``messages`` field, walks the list,
and for each assistant turn emits canonical atoms — sharing the
action-to-atom mapping with the swe-agent adapter so the two scaffolds
fingerprint into the same vocabulary.

Trace shape recap:

    {
      "messages": "[...JSON-string list of dicts...]",
      "instance_id": "...",
      "traj_id": "...",
      "model": "claude-3-7-sonnet-20250219",
      "resolved": true,
      "patch": "..."
    }

Each parsed message has ``role``, ``content``, ``agent``, ``message_type``;
assistant turns may additionally carry ``thought`` (reasoning), ``action``
(action name string), and ``tool_calls`` (structured function-call array
with ``function.name`` for fallback).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from procgrep.adapters.swe_agent import ATOM_MAP
from procgrep.canonicalize import register_adapter
from procgrep.types import ATOM_OTHER, ATOM_THINK, AtomSequence


def _parse_messages(raw: Any) -> list[Any]:
    """Return the messages list, parsing a JSON string if needed."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _action_name_from_message(msg: Mapping[str, Any]) -> str | None:
    """Extract the action name from an assistant turn.

    Prefers the explicit ``action`` field, falling back to the first
    ``tool_calls[].function.name`` when present.
    """
    action = msg.get("action")
    if isinstance(action, str) and action:
        return action
    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0]
        if isinstance(first, Mapping):
            fn = first.get("function")
            if isinstance(fn, Mapping):
                name = fn.get("name")
                if isinstance(name, str) and name:
                    return name
    return None


def swe_smith_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert one SWE-smith-trajectories row into canonical atoms.

    Args:
        record: A row from the SWE-smith-trajectories dataset. The
            relevant key is ``messages`` (a JSON-string list of dicts
            in chat format). Other keys (``instance_id``, ``traj_id``,
            ``model``, ``resolved``, ``patch``) are preserved as
            metadata by the canonicalization layer but ignored here.

    Returns:
        Ordered atom sequence. Each assistant turn contributes one
        action atom (mapped via the shared ATOM_MAP with the swe-agent
        adapter). An ATOM_THINK atom is prepended to a turn when its
        ``thought`` field carries non-empty reasoning text. Unknown
        action names fall through to ATOM_OTHER.
    """
    messages = _parse_messages(record.get("messages"))

    atoms: AtomSequence = []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        if msg.get("role") != "assistant":
            continue

        thought = msg.get("thought")
        if isinstance(thought, str) and thought.strip():
            atoms.append(ATOM_THINK)

        name = _action_name_from_message(msg)
        if name is not None:
            atoms.append(ATOM_MAP.get(name, ATOM_OTHER))

    return atoms


register_adapter("swe-smith", swe_smith_adapter, overwrite=True)


__all__ = ["swe_smith_adapter"]
