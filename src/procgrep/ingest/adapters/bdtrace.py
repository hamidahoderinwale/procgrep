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
command string, so the command goes through the same `_classify_terminal_command`
the other terminal adapters share, and an unrecognised command stays ``other``
rather than inflating a specific atom.

Atom mapping::

    prompt                             -> prompt_ai
    edit, code_change                  -> edit
    read                               -> read_file
    search                             -> search_repo
    test                               -> run_test
    run, by classified command         -> run_test / version_control / package /
                                          lint / search_repo / read_file / run_code
    run, command matching no verb      -> other
    other / unrecognized event type    -> other

Pass ``--trace-id-field instance_id`` to ``procgrep canonicalize``. bdtrace
labels every record with its source as ``agent`` (overridable at import), so
grouping and cross-agent comparison work without further mapping.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import EventRule, field_in, make_event_adapter, register_adapter
from procgrep.ingest.adapters.claude_code import _classify_terminal_command
from procgrep.types import (
    ATOM_EDIT,
    ATOM_LINT,
    ATOM_OTHER,
    ATOM_PACKAGE,
    ATOM_PROMPT_AI,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_VERSION_CONTROL,
)

# bdtrace types every shell call `run` and keeps the command, so the command is
# classified here with the same shared classifier the other terminal adapters
# use. Rolling a private one made every non-VCS command collapse to run_code,
# which inflates that atom and puts a structural zero under package and lint --
# fatal for the cross-agent comparison this adapter exists to enable.
_SUBKIND_ATOM = {
    "bash_test": ATOM_RUN_TEST,
    "bash_vcs": ATOM_VERSION_CONTROL,
    "bash_package": ATOM_PACKAGE,
    "bash_lint": ATOM_LINT,
    "bash_search": ATOM_SEARCH_REPO,
    "bash_read": ATOM_READ_FILE,
    "bash_run": ATOM_RUN_CODE,
}


def _flatten(event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Lift the command out of ``details`` so the rules stay flat field tests."""
    details = event.get("details")
    command = details.get("command", "") if isinstance(details, Mapping) else ""
    return {**event, "command": command if isinstance(command, str) else ""}


def _run_subkind(event: Mapping[str, Any]) -> str | None:
    if str(event.get("type", "")).lower() != "run":
        return None
    return _classify_terminal_command(str(event.get("command", "")))


def _run_is(subkind: str):
    """Predicate: a `run` event whose command classifies as this sub-kind."""

    def predicate(event: Mapping[str, Any]) -> bool:
        return _run_subkind(event) == subkind

    return predicate


def _run_unclassified(event: Mapping[str, Any]) -> bool:
    """A command no verb matched. It stays `other` rather than inflating a
    specific atom, the invariant the claude_code classifier is built around."""
    return _run_subkind(event) == "bash"


RULES = (
    EventRule(match=field_in("type", {"prompt"}), atoms=(ATOM_PROMPT_AI,)),
    EventRule(match=field_in("type", {"edit", "code_change"}), atoms=(ATOM_EDIT,)),
    EventRule(match=field_in("type", {"read"}), atoms=(ATOM_READ_FILE,)),
    EventRule(match=field_in("type", {"search"}), atoms=(ATOM_SEARCH_REPO,)),
    EventRule(match=field_in("type", {"test"}), atoms=(ATOM_RUN_TEST,)),
    *(EventRule(match=_run_is(subkind), atoms=(atom,)) for subkind, atom in _SUBKIND_ATOM.items()),
    EventRule(match=_run_unclassified, atoms=(ATOM_OTHER,)),
)

bdtrace_adapter = make_event_adapter(
    rules=RULES, events_path="events", default_atom=ATOM_OTHER, normalize=_flatten
)


def _register() -> None:
    register_adapter("bdtrace", bdtrace_adapter, overwrite=True)


_register()


__all__ = ["RULES", "bdtrace_adapter"]
