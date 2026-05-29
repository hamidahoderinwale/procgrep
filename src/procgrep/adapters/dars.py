"""DARS trace adapter.

Records carry an ``actions`` list of tool-call dicts with a ``tool``
name and optional ``thought``. Maps tool names through `ATOM_MAP` and
prepends ``ATOM_THINK`` on non-empty thoughts.
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
