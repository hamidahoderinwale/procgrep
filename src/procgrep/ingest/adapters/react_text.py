"""ReAct text-trajectory adapter.

Some datasets serialize actions as natural-language ReAct text rather than
structured ``tool_calls``: each assistant (or ``"ai"``) turn holds a thought
plus one or more fenced command blocks, e.g.::

    THOUGHT: explore the repo
    ```bash
    find . -name "*.py" | grep metadata
    ```

Covers nebius/SWE-agent-trajectories (role ``"ai"``, field ``text``) and
Kwai-Klear/...mini_swe_agent_plus (role ``assistant``, field ``content``).
Each fenced block's leading command is classified to a canonical atom; every
assistant turn emits ``THINK`` for its reasoning. Degrades gracefully.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import register_adapter
from procgrep.ingest.adapters.mini_swe_agent import _classify_command
from procgrep.types import ATOM_THINK, AtomSequence

ASSISTANT_ROLES = frozenset({"assistant", "ai"})
_FENCE = re.compile(r"```(?:bash|sh|shell|console|python|py)?\s*\n(.*?)```", re.DOTALL)


def _turn_text(turn: Mapping[str, Any]) -> str:
    for key in ("content", "text"):
        value = turn.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _commands(text: str) -> list[str]:
    """Leading command line of each fenced block in ``text``."""
    cmds: list[str] = []
    for block in _FENCE.findall(text):
        for line in block.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                cmds.append(stripped)
                break
    return cmds


def react_text_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert a ReAct-text ``messages`` record into canonical atoms."""
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
        atoms.append(ATOM_THINK)
        for command in _commands(text):
            atoms.append(_classify_command(command))
    return atoms


def _register() -> None:
    register_adapter("react-text", react_text_adapter, overwrite=True)


_register()


__all__ = ["react_text_adapter"]
