"""Core data types for `procgrep`.

This module defines the small, stable vocabulary of types that the rest
of the package consumes. Keeping these together in one file lets a
reader understand the data model in one sitting before encountering
the algorithms that operate on it.

Three concepts:

1. An `Atom` is a single canonical action label. The canonical alphabet
   is small and curated; the constants `ATOM_LOCALIZE`, `ATOM_EDIT`,
   etc. enumerate the alphabet used by the built-in scaffold adapters.
   Users may register additional atoms via custom adapters.

2. An `AtomSequence` is the procedural shape of one trajectory: an
   ordered list of atoms in the order they were emitted by the agent.

3. A `Trace` is one trajectory plus its identity (trace id, agent
   name), an optional grouping label used for cross-group comparison,
   and a free-form metadata dictionary.

A `TraceAdapter` is the function shape that converts a raw, scaffold-
specific trace record into an `AtomSequence`. Adapters live in
`procgrep.canonicalize` and are registered by name there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TypeAlias

Atom: TypeAlias = str
"""A single canonical action label, e.g. ``"edit"`` or ``"run_test"``."""

AtomSequence: TypeAlias = list[Atom]
"""The ordered procedural shape of one trajectory."""

TraceAdapter: TypeAlias = Callable[[Mapping[str, object]], AtomSequence]
"""Function that maps a raw scaffold-specific trace record to atoms."""


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
"""The default canonical atom alphabet used by built-in adapters."""

MOTIF_SEPARATOR: str = "▁"
"""Separator used to glue atoms into multi-atom BPE motifs (``"edit▁run_test"``)."""


@dataclass(frozen=True)
class Trace:
    """One agent trajectory, canonicalized into atoms.

    Attributes:
        trace_id: Stable identifier unique within a corpus.
        agent: Name of the agent that produced the trajectory.
        atoms: The canonical procedural shape (ordered atom list).
        group: Optional grouping label used by JSD and the probe
            (paradigm x scaffold cell, controlled-eval arm, etc.).
            If omitted, downstream commands fall back to ``agent``
            as the grouping variable.
        metadata: Free-form, JSON-serializable metadata. Reserved
            for project-specific fields (instance id, success flag,
            wallclock, etc.); `procgrep` itself does not interpret
            this dictionary.
    """

    trace_id: str
    agent: str
    atoms: AtomSequence
    group: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def grouping(self) -> str:
        """Return ``group`` if set, otherwise ``agent``."""
        return self.group if self.group is not None else self.agent
