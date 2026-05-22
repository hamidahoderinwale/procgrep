"""Moatless trace adapter.

Moatless (https://github.com/aorwall/moatless-tools) emits traces
whose action steps carry an ``action`` name (CamelCase tool labels
like ``FindCode``, ``RequestCodeChange``) and an optional ``thoughts``
string. This adapter maps the action name through a synonym table to
a canonical atom and prepends ``ATOM_THINK`` when a step has non-
empty thoughts text.
"""

from __future__ import annotations

from procgrep.canonicalize import make_action_adapter, register_adapter
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_EDIT,
    ATOM_LOCALIZE,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    Atom,
)

ATOM_MAP: dict[str, Atom] = {
    "FindCode": ATOM_LOCALIZE,
    "FindClass": ATOM_LOCALIZE,
    "FindFunction": ATOM_LOCALIZE,
    "SemanticSearch": ATOM_SEARCH_REPO,
    "ViewCode": ATOM_READ_FILE,
    "RequestCodeChange": ATOM_EDIT,
    "StringReplace": ATOM_EDIT,
    "CreateFile": ATOM_CREATE_FILE,
    "RunTests": ATOM_RUN_TEST,
    "Finish": ATOM_SUBMIT,
}


adapter = make_action_adapter(
    action_field="action",
    atom_map=ATOM_MAP,
    thought_field="thoughts",
)


register_adapter("moatless", adapter, overwrite=True)


__all__ = ["ATOM_MAP", "adapter"]
