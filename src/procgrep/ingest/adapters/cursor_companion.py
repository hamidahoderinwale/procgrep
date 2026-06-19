"""Cursor companion trace adapter.

Converts traces exported from the cursor-telemetry companion service
(https://github.com/hamidahoderinwale/cursor-telemetry) into procgrep atoms.
Run that companion alongside Cursor and hit its ``/api/export/procgrep``
endpoint to capture your own human+AI session traces. cursor-companion is a
separate project woven in only as an ingest adapter -- an exemplar trace
source, not part of procgrep's core.

The companion captures human+AI Cursor sessions: AI prompts, code edits,
terminal commands, and file reads/searches. This adapter is defined
declaratively as feature-based rules over `make_event_adapter`, so a single
export event decomposes into the atoms its fields imply. A prompt turn that
also edited code becomes ``prompt_ai`` then ``edit`` -- a decomposition a
one-type-one-atom mapping cannot express, and the reason the mapping is by
event *features* rather than a pre-classified type string.

Trace shape produced by the companion's ``/api/export/procgrep`` endpoint::

    {
      "trace_id": "session-abc123",
      "agent": "developer-fingerprint",
      "group": "before_ai" | "after_ai",      # optional, for longitudinal splits
      "events": [
        {"type": "prompt_with_edit", "lines_added": 12, "context_files": ["a.ts"]},
        {"type": "prompt",           "text": "refactor this"},
        {"type": "terminal",         "command": "npm test"},
        {"type": "file_open",        "file_path": "src/api.ts"},
        ...
      ]
    }

Atom mapping (by event features, additive, in order):

    file_search / search / grep            -> search_repo
    file_open / file_read / context_files  -> read_file
    prompt / *_prompt / prompt_with_edit   -> prompt_ai   (new atom, human->AI turn)
    edit types / prompt_with_edit /
      lines added or removed                -> edit
    terminal / command_run                  -> run_test    (nearest canonical proxy)
    (no rule matches)                        -> other

Design decisions:

- ``prompt_ai`` is a first-class atom, not aliased to ``think``: ``think`` is
  the agent reasoning before an action; ``prompt_ai`` is the human handing off
  to the AI. Keeping them distinct lets fingerprinting separate AI-heavy from
  manual work.

- Edit-ness is derived from event *fields* (``lines_added`` / ``lines_removed``)
  as well as the type, so the exporter need not pre-classify a turn as an edit;
  a composite ``prompt_with_edit`` turn yields ``prompt_ai`` then ``edit``.

- Terminal commands map to ``run_test`` because in practice they are
  predominantly test/build commands; this matches the SWE-agent convention and
  keeps the alphabet shared.

- A nested ``details`` blob is flattened into top-level fields by ``_normalize``
  before the generic rules run, keeping the companion's event shape out of the
  shared mapping machinery.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import (
    EventRule,
    any_of,
    field_in,
    field_truthy,
    make_event_adapter,
    register_adapter,
)
from procgrep.types import (
    ATOM_EDIT,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
)

_EDIT_TYPES = {"code_change", "file_change", "entry_created", "edit", "file_save"}
_PROMPT_TYPES = {"prompt", "ai_prompt", "llm_prompt", "prompt_with_edit"}
_READ_TYPES = {"file_open", "file_read"}
_SEARCH_TYPES = {"file_search", "search", "grep"}
_TERMINAL_TYPES = {"terminal", "terminal_command", "command_run"}

# Feature rules, evaluated in order; a composite event fires several of them.
# Ordered exploration -> handoff -> generation, so an edit-with-context prompt
# reads as read_file, prompt_ai, edit. prompt_with_edit triggers both the
# prompt and the edit rule, so it decomposes even when line counts are absent.
CURSOR_RULES: tuple[EventRule, ...] = (
    EventRule(field_in("type", _SEARCH_TYPES), (ATOM_SEARCH_REPO,)),
    EventRule(
        any_of(field_in("type", _READ_TYPES), field_truthy("context_files")),
        (ATOM_READ_FILE,),
    ),
    EventRule(field_in("type", _PROMPT_TYPES), (ATOM_PROMPT_AI,)),
    EventRule(
        any_of(
            field_in("type", _EDIT_TYPES | {"prompt_with_edit"}),
            field_truthy("lines_added", "lines_removed"),
        ),
        (ATOM_EDIT,),
    ),
    EventRule(field_in("type", _TERMINAL_TYPES), (ATOM_RUN_TEST,)),
)


def _parse_details(raw: Any) -> dict[str, Any]:
    """Parse an event's ``details`` into a dict; tolerate JSON strings and junk."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Flatten a nested ``details`` blob into top-level fields.

    Top-level non-empty values win; ``details`` fills the gaps, so an event
    carrying only ``details.type`` still classifies. This keeps the companion's
    nested shape out of the generic rule machinery.
    """
    merged: dict[str, Any] = dict(_parse_details(event.get("details")))
    for key, value in event.items():
        if value not in (None, ""):
            merged[key] = value
    return merged


cursor_companion_adapter = make_event_adapter(rules=CURSOR_RULES, normalize=_normalize)
register_adapter("cursor-companion", cursor_companion_adapter, overwrite=True)

__all__ = ["ATOM_PROMPT_AI", "CURSOR_RULES", "cursor_companion_adapter"]
