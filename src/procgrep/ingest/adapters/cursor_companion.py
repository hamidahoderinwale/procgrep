"""Cursor companion trace adapter.

Converts traces exported from the bidirect-align-dev companion service
(https://github.com/Taste-AI/bidirect-align-dev) into procgrep atoms.

The companion captures human+AI sessions from Cursor IDE: code edits,
prompts sent to the AI, terminal commands, and file reads. This adapter
maps those event types onto the canonical atom vocabulary, extending it
with ``prompt_ai`` for AI prompt events — the one action type that has
no equivalent in autonomous-agent traces.

Trace shape produced by the companion's ``/api/export/procgrep`` endpoint::

    {
      "trace_id": "session-abc123",
      "agent": "developer-fingerprint",     # workspace hash or user id
      "group": "before_ai" | "after_ai",    # optional, for longitudinal splits
      "events": [
        {"type": "code_change",  "file_path": "src/utils.ts", "timestamp": 1234567890, "prompt_id": "p1"},
        {"type": "prompt",       "text": "refactor this function", "timestamp": 1234567891},
        {"type": "terminal",     "command": "npm test", "timestamp": 1234567892},
        {"type": "file_open",    "file_path": "src/api.ts", "timestamp": 1234567893},
        ...
      ]
    }

Atom mapping:

    Companion event type            → Atom
    ─────────────────────────────────────────────────────
    code_change / file_change /
    entry_created / edit / file_save → edit
    prompt / ai_prompt / llm_prompt  → prompt_ai  (new atom, human→AI turn)
    terminal / terminal_command /
    command_run                      → run_test    (nearest canonical proxy)
    file_open / file_read            → read_file
    file_search / search / grep      → search_repo
    (unknown)                        → other

Design decisions:

- ``prompt_ai`` is registered as a first-class atom rather than aliased to
  ``think`` because it is directionally opposite: ``think`` is the agent
  reasoning before an action; ``prompt_ai`` is the human handing off to the
  AI. Keeping them distinct lets fingerprinting separate AI-heavy from
  manual sessions.

- Terminal commands are mapped to ``run_test`` because in practice they are
  predominantly test/build commands. This matches the SWE-agent convention
  and keeps the alphabet shared.

- ``file_save`` without a content diff maps to ``edit`` (intent is clear
  even if the change is zero-byte) rather than ``other``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import register_adapter
from procgrep.types import (
    ATOM_EDIT,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    Atom,
    AtomSequence,
)

ATOM_PROMPT_AI: Atom = "prompt_ai"
"""Human→AI turn — the developer sends a prompt to the Cursor AI."""

_EVENT_TYPE_ATOM: dict[str, Atom] = {
    # edits
    "code_change": ATOM_EDIT,
    "file_change": ATOM_EDIT,
    "entry_created": ATOM_EDIT,
    "edit": ATOM_EDIT,
    "file_save": ATOM_EDIT,
    # AI prompts
    "prompt": ATOM_PROMPT_AI,
    "ai_prompt": ATOM_PROMPT_AI,
    "llm_prompt": ATOM_PROMPT_AI,
    # terminal
    "terminal": ATOM_RUN_TEST,
    "terminal_command": ATOM_RUN_TEST,
    "command_run": ATOM_RUN_TEST,
    # file reads / navigation
    "file_open": ATOM_READ_FILE,
    "file_read": ATOM_READ_FILE,
    # search
    "file_search": ATOM_SEARCH_REPO,
    "search": ATOM_SEARCH_REPO,
    "grep": ATOM_SEARCH_REPO,
}


def _parse_details(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def cursor_companion_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Convert one companion session record into an atom sequence.

    Events are emitted in their stored order (the companion persists
    events in arrival order; callers should sort by timestamp before
    exporting if strict chronology is required).
    """
    events = record.get("events") or []
    if not isinstance(events, list):
        return []

    atoms: AtomSequence = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        raw_type = str(event.get("type") or "").lower()
        details = _parse_details(event.get("details"))
        # details.type can override top-level type for nested events
        if not raw_type:
            raw_type = str(details.get("type") or "").lower()
        atoms.append(_EVENT_TYPE_ATOM.get(raw_type, ATOM_OTHER))

    return atoms


register_adapter("cursor-companion", cursor_companion_adapter, overwrite=True)

__all__ = ["ATOM_PROMPT_AI", "cursor_companion_adapter"]
