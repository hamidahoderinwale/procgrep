"""SWE-agent trace adapter.

SWE-agent (https://github.com/SWE-agent/SWE-agent) emits traces whose
top-level record carries a list of action steps under ``actions``.
Each step has an ``action`` string naming the tool call and an
optional ``thought`` string carrying the model's reasoning. This
adapter maps the action name through a synonym table to a canonical
atom and prepends ``ATOM_THINK`` when a step has non-empty thought
text.
"""

from __future__ import annotations

from procgrep.canonicalize import make_action_adapter, register_adapter
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_DELETE_FILE,
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    Atom,
)

ATOM_MAP: dict[str, Atom] = {
    "edit": ATOM_EDIT,
    "str_replace": ATOM_EDIT,
    "str_replace_editor": ATOM_EDIT,
    "create": ATOM_CREATE_FILE,
    "delete": ATOM_DELETE_FILE,
    "open": ATOM_READ_FILE,
    "goto": ATOM_READ_FILE,
    "view": ATOM_READ_FILE,
    "scroll_down": ATOM_READ_FILE,
    "scroll_up": ATOM_READ_FILE,
    "search_dir": ATOM_SEARCH_REPO,
    "search_file": ATOM_SEARCH_REPO,
    "find_file": ATOM_SEARCH_REPO,
    "grep": ATOM_SEARCH_REPO,
    "run_test": ATOM_RUN_TEST,
    "pytest": ATOM_RUN_TEST,
    "submit": ATOM_SUBMIT,
    "exit": ATOM_SUBMIT,
}


adapter = make_action_adapter(
    action_field="action",
    atom_map=ATOM_MAP,
    thought_field="thought",
)


register_adapter("swe-agent", adapter, overwrite=True)


__all__ = ["ATOM_MAP", "adapter"]
