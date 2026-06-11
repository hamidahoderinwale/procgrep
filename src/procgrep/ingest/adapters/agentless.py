"""Agentless trace adapter.

Records carry an ``actions`` list of phase dicts with a ``phase``
name and optional ``reasoning``. Maps phase names through `ATOM_MAP`
and prepends ``ATOM_THINK`` on non-empty reasoning. See
https://github.com/OpenAutoCoder/Agentless.
"""

from __future__ import annotations

from procgrep.canonicalize import make_action_adapter, register_adapter
from procgrep.types import ATOM_EDIT, ATOM_LOCALIZE, ATOM_RUN_TEST, ATOM_SUBMIT, Atom

ATOM_MAP: dict[str, Atom] = {
    "fault_localization": ATOM_LOCALIZE,
    "file_localization": ATOM_LOCALIZE,
    "line_localization": ATOM_LOCALIZE,
    "repair": ATOM_EDIT,
    "patch_generation": ATOM_EDIT,
    "regression_test": ATOM_RUN_TEST,
    "reproduction_test": ATOM_RUN_TEST,
    "submit": ATOM_SUBMIT,
}


adapter = make_action_adapter(
    action_field="phase",
    atom_map=ATOM_MAP,
    thought_field="reasoning",
)


register_adapter("agentless", adapter, overwrite=True)


__all__ = ["ATOM_MAP", "adapter"]
