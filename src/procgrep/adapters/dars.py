"""DARS trace adapter.

DARS-style traces carry a list of tool-call steps under ``actions``,
each with a ``tool`` name and an optional ``thought`` string. This
adapter maps the tool name through a synonym table to a canonical
atom and prepends ``ATOM_THINK`` when a step has non-empty thought
text.
"""

from __future__ import annotations

from procgrep.canonicalize import make_action_adapter, register_adapter
from procgrep.types import (
    ATOM_EDIT,
    ATOM_ERROR,
    ATOM_LOCALIZE,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
    Atom,
)

ATOM_MAP: dict[str, Atom] = {
    "localize": ATOM_LOCALIZE,
    "read": ATOM_READ_FILE,
    "edit": ATOM_EDIT,
    "test": ATOM_RUN_TEST,
    "search": ATOM_SEARCH_REPO,
    "submit": ATOM_SUBMIT,
    "reason": ATOM_THINK,
    "error": ATOM_ERROR,
}


adapter = make_action_adapter(
    action_field="tool",
    atom_map=ATOM_MAP,
    thought_field="thought",
)


register_adapter("dars", adapter, overwrite=True)


__all__ = ["ATOM_MAP", "adapter"]
