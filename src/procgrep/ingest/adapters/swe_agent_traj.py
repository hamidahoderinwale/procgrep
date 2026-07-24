"""SWE-agent leaderboard ``.traj`` adapter.

SWE-bench leaderboard submissions store one ``.traj`` per instance: a dict with
a ``trajectory`` list of step dicts. Unlike the lighter ``swe-agent`` records
(a clean ``actions`` list of named actions), each step here carries a full
command string in ``action`` (e.g. ``str_replace_editor view ...``,
``find ... | grep ...``), so this adapter classifies the command rather than
looking up a name. The ``str_replace_editor`` tool carries its operation in the
second token; everything else is a shell command routed through the shared
terminal classifier. A non-empty ``thought`` prepends ATOM_THINK, matching the
``swe-agent`` adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from procgrep.canonicalize import register_adapter
from procgrep.ingest.adapters.claude_code import _classify_terminal_command
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_EDIT,
    ATOM_LINT,
    ATOM_OTHER,
    ATOM_PACKAGE,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
    ATOM_VERSION_CONTROL,
    Atom,
    AtomSequence,
)

# _classify_terminal_command returns an intermediate bash sub-kind; map it onto
# the canonical alphabet so this adapter emits the same atoms as every other one
# (the shared _BASH_ATOM omits bash_read, so we keep a complete map here).
_KIND_ATOM: dict[str, Atom] = {
    "bash_read": ATOM_READ_FILE,
    "bash_search": ATOM_SEARCH_REPO,
    "bash_test": ATOM_RUN_TEST,
    "bash_vcs": ATOM_VERSION_CONTROL,
    "bash_package": ATOM_PACKAGE,
    "bash_lint": ATOM_LINT,
    "bash_run": ATOM_RUN_CODE,
    "bash": ATOM_OTHER,
}

# str_replace_editor operation -> atom; the editor tool names its operation in
# the second token of the action string.
_EDITOR_OP: dict[str, Atom] = {
    "view": ATOM_READ_FILE,
    "open": ATOM_READ_FILE,
    "goto": ATOM_READ_FILE,
    "scroll_down": ATOM_READ_FILE,
    "scroll_up": ATOM_READ_FILE,
    "str_replace": ATOM_EDIT,
    "insert": ATOM_EDIT,
    "create": ATOM_CREATE_FILE,
}
_EDITOR_TOOLS = {"str_replace_editor", "str_replace_based_edit_tool"}
_SUBMIT_VERBS = {"submit", "exit", "exit_cost", "exit_context", "exit_format"}


def _classify_action(action: str) -> Atom:
    toks = action.strip().split()
    if not toks:
        return "other"
    head = toks[0]
    if head in _EDITOR_TOOLS:
        op = toks[1] if len(toks) > 1 else ""
        return _EDITOR_OP.get(op, ATOM_EDIT)
    if head in _SUBMIT_VERBS:
        return ATOM_SUBMIT
    # Otherwise it is a shell command; classify the verb, then map the bash
    # sub-kind onto the canonical alphabet.
    return _KIND_ATOM.get(_classify_terminal_command(action), ATOM_OTHER)


def swe_agent_traj_adapter(record: Mapping[str, Any]) -> AtomSequence:
    atoms: AtomSequence = []
    for step in record.get("trajectory") or []:
        if not isinstance(step, Mapping):
            continue
        if str(step.get("thought") or "").strip():
            atoms.append(ATOM_THINK)
        action = step.get("action")
        if isinstance(action, str) and action.strip():
            atoms.append(_classify_action(action))
    return atoms


register_adapter("swe-agent-traj", swe_agent_traj_adapter, overwrite=True)

__all__ = ["swe_agent_traj_adapter"]
