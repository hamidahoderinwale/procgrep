"""bdtrace standardized-record adapter.

`bdtrace <https://github.com/hamidahoderinwale/bdtrace>`_ pulls traces out of
local agent stores (Claude Code, Cursor, SWE-agent, OpenHands) and emits one
normalized JSONL record per session, so a single adapter covers every source it
supports. Its records also survive anonymization and compressed re-export,
which is how a trace someone else collected usually arrives.

Expected record shape (one record == one session or benchmark instance)::

    {"instance_id": "claude-<uuid>", "repo": null, "base_commit": null,
     "events": [{"type": "run", "timestamp": "...",
                 "details": {"tool": "Bash", "command": "pytest -q"}}, ...]}

bdtrace has already classified each event into a closed taxonomy, so the rules
below map that taxonomy rather than re-deriving it from tool names. The one
exception is ``run``: bdtrace types every shell command as ``run`` and keeps the
command string, so the command is inspected here to split version control off
from ordinary execution, matching what the claude_code adapter does natively.

Atom mapping::

    prompt                             -> prompt_ai
    edit, code_change                  -> edit
    read                               -> read_file
    search                             -> search_repo
    test                               -> run_test
    run, git/gh/hg/svn command         -> version_control
    run, anything else                 -> run_code
    other / unrecognized               -> other

Pass ``--trace-id-field instance_id`` to ``procgrep canonicalize``. bdtrace
labels every record with its source as ``agent`` (overridable at import), so
grouping and cross-agent comparison work without further mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import EventRule, field_in, make_event_adapter, register_adapter
from procgrep.types import (
    ATOM_EDIT,
    ATOM_OTHER,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_VERSION_CONTROL,
)

VCS_PREFIXES = ("git ", "gh ", "hg ", "svn ")


def _flatten(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Lift the command out of ``details`` so the rules stay flat field tests."""
    details = event.get("details")
    command = details.get("command", "") if isinstance(details, Mapping) else ""
    return {**event, "command": command if isinstance(command, str) else ""}


def _is_vcs(event: Mapping[str, Any]) -> bool:
    if str(event.get("type", "")).lower() != "run":
        return False
    return str(event.get("command", "")).strip().lower().startswith(VCS_PREFIXES)


def _is_plain_run(event: Mapping[str, Any]) -> bool:
    return str(event.get("type", "")).lower() == "run" and not _is_vcs(event)


RULES = (
    EventRule(match=field_in("type", {"prompt"}), atoms=(ATOM_PROMPT_AI,)),
    EventRule(match=field_in("type", {"edit", "code_change"}), atoms=(ATOM_EDIT,)),
    EventRule(match=field_in("type", {"read"}), atoms=(ATOM_READ_FILE,)),
    EventRule(match=field_in("type", {"search"}), atoms=(ATOM_SEARCH_REPO,)),
    EventRule(match=field_in("type", {"test"}), atoms=(ATOM_RUN_TEST,)),
    EventRule(match=_is_vcs, atoms=(ATOM_VERSION_CONTROL,)),
    EventRule(match=_is_plain_run, atoms=(ATOM_RUN_CODE,)),
)

bdtrace_adapter = make_event_adapter(
    rules=RULES, events_path="events", default_atom=ATOM_OTHER, normalize=_flatten
)


def _register() -> None:
    register_adapter("bdtrace", bdtrace_adapter, overwrite=True)


_register()


__all__ = ["RULES", "VCS_PREFIXES", "bdtrace_adapter"]
