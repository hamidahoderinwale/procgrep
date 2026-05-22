"""Convert heterogeneous scaffold-specific traces into canonical atoms.

The canonical alphabet (`procgrep.types.CANONICAL_ATOMS`) is the
shared substrate that lets fingerprints from different scaffolds be
compared. Each scaffold ships an *adapter*: a callable that takes
one raw trace record and returns an `AtomSequence`. Built-in
adapters live under `procgrep.adapters` and self-register at import
time; custom-scaffold adapters can register through this module's
public API.

The module exposes:

* `register_adapter(name, adapter)`: add an adapter to the registry.
* `get_adapter(name)`: look an adapter up by name.
* `list_adapters()`: enumerate the registered adapter names.
* `canonicalize(traces, adapter=...)`: apply an adapter (by name or
  callable) to a corpus of raw records, returning canonical `Trace`
  objects.
* `make_action_adapter(...)`: factory for the common
  "list-of-actions" trace shape; the four built-in agent-action
  adapters are built with this.
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

    Args:
        name: Lookup key (lowercase, hyphen-separated convention).
        adapter: Callable mapping one raw record to an `AtomSequence`.
        overwrite: If True, replace an existing entry with the same
            name; otherwise raise `ValueError` on conflict.
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
    """Return the names of all registered adapters in sorted order."""
    return sorted(_ADAPTERS)


def canonicalize(
    traces: Iterable[Mapping[str, Any]],
    *,
    adapter: str | TraceAdapter,
    trace_id_field: str = "trace_id",
    agent_field: str = "agent",
    group_field: str | None = "group",
) -> list[Trace]:
    """Canonicalize a corpus of raw trace records into `Trace` objects.

    Args:
        traces: Iterable of dict-like records. Each record must carry
            at minimum the trace id and agent fields named below; the
            adapter pulls action information from the rest.
        adapter: Adapter name (string) to look up in the registry,
            or a callable conforming to `TraceAdapter`.
        trace_id_field: Key in each record holding the trace id.
        agent_field: Key in each record holding the agent name.
        group_field: Optional key holding the grouping label. If the
            field is absent in a record, the resulting `Trace.group`
            is None.

    Returns:
        A list of `Trace` objects in the same order as the input.
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
    """Build a TraceAdapter for the common "list-of-actions" shape.

    Many scaffold trace formats share a structure: the record contains
    a list of action steps, each carrying a name and optional thought
    text. This factory builds an adapter that walks the list, maps
    each action's name through ``atom_map`` to a canonical atom, and
    optionally prepends an ``ATOM_THINK`` atom whenever a step
    carries non-empty thought text.

    Args:
        action_field: Within each step dict, the key holding the
            action name.
        atom_map: Synonym table from action name to canonical atom.
            Names not in the map fall through to ``default_atom``.
        thought_field: If set, steps with non-empty values at this
            key emit an ``ATOM_THINK`` atom before the action atom.
        actions_path: Key in the outer record holding the list of
            step dicts. Defaults to ``"actions"``.
        default_atom: Atom emitted when a step's action name is not
            in ``atom_map``.

    Returns:
        A callable suitable for `register_adapter`.
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
