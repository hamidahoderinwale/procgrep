"""Core data types for `procgrep`.

Defines `Atom` (a canonical action label), `AtomSequence` (the ordered
atom list for one trajectory), `Trace` (one trajectory plus identity
and grouping metadata), and `TraceAdapter` (the function shape that
converts a raw trace record to atoms). Built-in atoms are enumerated
as `ATOM_*` constants.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

Atom: TypeAlias = str
"""A canonical action label, e.g. ``"edit"`` or ``"run_test"``."""

AtomSequence: TypeAlias = list[Atom]
"""Ordered atoms for one trajectory."""

TraceAdapter: TypeAlias = Callable[[Mapping[str, object]], AtomSequence]
"""Maps a raw trace record to an atom sequence."""


ATOM_LOCALIZE: Atom = "localize"
ATOM_READ_FILE: Atom = "read_file"
ATOM_EDIT: Atom = "edit"
ATOM_RUN_TEST: Atom = "run_test"
ATOM_SEARCH_REPO: Atom = "search_repo"
ATOM_CREATE_FILE: Atom = "create_file"
ATOM_DELETE_FILE: Atom = "delete_file"
ATOM_SUBMIT: Atom = "submit"
ATOM_THINK: Atom = "think"
ATOM_ERROR: Atom = "error"
ATOM_OTHER: Atom = "other"

CANONICAL_ATOMS: frozenset[Atom] = frozenset(
    {
        ATOM_LOCALIZE,
        ATOM_READ_FILE,
        ATOM_EDIT,
        ATOM_RUN_TEST,
        ATOM_SEARCH_REPO,
        ATOM_CREATE_FILE,
        ATOM_DELETE_FILE,
        ATOM_SUBMIT,
        ATOM_THINK,
        ATOM_ERROR,
        ATOM_OTHER,
    }
)
"""Default atom alphabet used by built-in adapters."""

PROCEDURE_SEPARATOR: str = "▁"
"""Glues atoms into multi-atom BPE procedures (``"edit▁run_test"``)."""


@dataclass(frozen=True)
class Trace:
    """One agent trajectory, canonicalized into atoms.

    Attributes:
        trace_id: Stable identifier within a corpus.
        agent: Agent that produced the trajectory.
        atoms: Ordered atom list.
        group: Grouping label for JSD and probes. Falls back to
            ``agent`` when unset.
        metadata: JSON-serializable, project-specific. Not interpreted
            by `procgrep`.
    """

    trace_id: str
    agent: str
    atoms: AtomSequence
    group: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def grouping(self) -> str:
        """Return ``group`` if set, otherwise ``agent``."""
        return self.group if self.group is not None else self.agent
