"""Convert scaffold-specific traces into canonical atoms.

Each scaffold ships an adapter (a callable from raw record to
`AtomSequence`). Built-in adapters live under `procgrep.ingest.adapters` and
self-register at import. Public API: `register_adapter`, `get_adapter`,
`list_adapters`, `canonicalize`, and `make_action_adapter` for the
common list-of-actions shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from procgrep.types import (
    ATOM_OTHER,
    ATOM_THINK,
    Atom,
    AtomSequence,
    Trace,
    TraceAdapter,
)

_ADAPTERS: dict[str, TraceAdapter] = {}


def register_adapter(name: str, adapter: TraceAdapter, *, overwrite: bool = False) -> None:
    """Register an adapter under ``name``.

    Raises:
        ValueError: If ``name`` is already registered and
            ``overwrite`` is False.
    """
    if not overwrite and name in _ADAPTERS:
        raise ValueError(f"adapter already registered: {name!r}")
    _ADAPTERS[name] = adapter


def get_adapter(name: str) -> TraceAdapter:
    """Return the adapter registered under ``name``."""
    if name not in _ADAPTERS:
        known = ", ".join(sorted(_ADAPTERS)) or "<none registered>"
        raise KeyError(f"no adapter named {name!r}; known adapters: {known}")
    return _ADAPTERS[name]


def list_adapters() -> list[str]:
    """Return registered adapter names, sorted."""
    return sorted(_ADAPTERS)


def canonicalize(
    traces: Iterable[Mapping[str, Any]],
    *,
    adapter: str | TraceAdapter,
    trace_id_field: str = "trace_id",
    agent_field: str = "agent",
    group_field: str | None = "group",
) -> list[Trace]:
    """Canonicalize raw trace records into `Trace` objects.

    Args:
        adapter: Registered adapter name or a `TraceAdapter` callable.
        group_field: Records missing this key produce ``Trace.group=None``.
    """
    fn: TraceAdapter = get_adapter(adapter) if isinstance(adapter, str) else adapter
    out: list[Trace] = []
    for record in traces:
        atoms = fn(record)
        trace = Trace(
            trace_id=str(record[trace_id_field]),
            agent=str(record[agent_field]),
            atoms=list(atoms),
            group=(
                None
                if group_field is None or group_field not in record
                else str(record[group_field])
            ),
            metadata={k: v for k, v in record.items() if k not in {trace_id_field, agent_field}},
        )
        out.append(trace)
    return out


def make_action_adapter(
    *,
    action_field: str,
    atom_map: Mapping[str, Atom],
    thought_field: str | None = None,
    actions_path: str = "actions",
    default_atom: Atom = ATOM_OTHER,
) -> TraceAdapter:
    """Build a TraceAdapter for the list-of-actions trace shape.

    The record holds a list of step dicts; each step carries an action
    name and optional thought text. The adapter maps each action name
    through ``atom_map``, prepending an ``ATOM_THINK`` atom when
    ``thought_field`` is set and non-empty for that step.

    Args:
        atom_map: Action name to canonical atom. Misses fall through
            to ``default_atom``.
        thought_field: When set, non-empty thought text emits
            ``ATOM_THINK`` before the action atom.
        actions_path: Key in the outer record holding the step list.
    """

    def adapter(record: Mapping[str, Any]) -> AtomSequence:
        steps = record.get(actions_path, [])
        if not isinstance(steps, list):
            raise TypeError(
                f"expected list at record[{actions_path!r}], got {type(steps).__name__}"
            )
        atoms: AtomSequence = []
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            if thought_field is not None:
                thought = step.get(thought_field)
                if isinstance(thought, str) and thought.strip():
                    atoms.append(ATOM_THINK)
            name = step.get(action_field)
            if name is None:
                atoms.append(default_atom)
            else:
                atoms.append(atom_map.get(str(name), default_atom))
        return atoms

    return adapter


__all__ = [
    "canonicalize",
    "get_adapter",
    "list_adapters",
    "make_action_adapter",
    "register_adapter",
]
