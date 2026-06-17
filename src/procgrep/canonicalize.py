"""Convert scaffold-specific traces into canonical atoms.

Each scaffold ships an adapter (a callable from raw record to
`AtomSequence`). Built-in adapters live under `procgrep.ingest.adapters` and
self-register at import. Public API: `register_adapter`, `get_adapter`,
`list_adapters`, `canonicalize`, `make_action_adapter` (the common
list-of-actions shape), and `make_event_adapter` (feature-based event shape).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EventRule:
    """One rule in a feature-based event mapping.

    When ``match`` holds for an event, the rule contributes ``atoms`` (in
    order) to that event's output. Rules compose additively, so a single
    composite event can decompose into several atoms.
    """

    match: Callable[[Mapping[str, Any]], bool]
    atoms: tuple[Atom, ...]


def field_in(field: str, values: Collection[str]) -> Callable[[Mapping[str, Any]], bool]:
    """Predicate: ``event[field]`` equals one of ``values``, case-insensitively."""
    lowered = {v.lower() for v in values}

    def predicate(event: Mapping[str, Any]) -> bool:
        return str(event.get(field, "")).lower() in lowered

    return predicate


def field_truthy(*fields: str) -> Callable[[Mapping[str, Any]], bool]:
    """Predicate: any of ``fields`` is present and truthy on the event."""

    def predicate(event: Mapping[str, Any]) -> bool:
        return any(event.get(f) for f in fields)

    return predicate


def any_of(
    *predicates: Callable[[Mapping[str, Any]], bool],
) -> Callable[[Mapping[str, Any]], bool]:
    """Combine predicates with logical OR."""

    def predicate(event: Mapping[str, Any]) -> bool:
        return any(p(event) for p in predicates)

    return predicate


def make_event_adapter(
    *,
    rules: Sequence[EventRule],
    events_path: str = "events",
    default_atom: Atom = ATOM_OTHER,
    dedupe_per_event: bool = True,
    normalize: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> TraceAdapter:
    """Build a TraceAdapter that derives atoms from event *features*.

    The record holds a list of event dicts at ``events_path``. For each event,
    every rule whose predicate holds fires, in order, and the rules' atoms are
    concatenated, so a single composite event (say, an AI prompt that also
    produced an edit) decomposes into several atoms. Within one event duplicate
    atoms are collapsed when ``dedupe_per_event`` is set, preserving first-seen
    order; an event matching no rule emits ``default_atom``.

    ``normalize`` optionally rewrites each event before the rules run -- where a
    source folds its own shape (e.g. a nested ``details`` blob) into flat
    fields, keeping that per-source so the rule machinery stays generic.

    A missing or non-list ``events_path`` yields an empty sequence rather than
    raising: this maps event streams, where a malformed event should not crash
    the run. That is the deliberate difference from `make_action_adapter`,
    which maps a single action name per step and is strict about its shape.

    This generalizes `make_action_adapter`: events map by field predicates
    rather than one action-name lookup, and one event may yield many atoms.
    """

    def adapter(record: Mapping[str, Any]) -> AtomSequence:
        events = record.get(events_path)
        if not isinstance(events, list):
            return []
        atoms: AtomSequence = []
        for event in events:
            if not isinstance(event, Mapping):
                continue
            normalized = normalize(event) if normalize is not None else event
            emitted: list[Atom] = []
            for rule in rules:
                if rule.match(normalized):
                    emitted.extend(rule.atoms)
            if dedupe_per_event:
                deduped: list[Atom] = []
                for atom in emitted:
                    if atom not in deduped:
                        deduped.append(atom)
                emitted = deduped
            atoms.extend(emitted or [default_atom])
        return atoms

    return adapter


__all__ = [
    "EventRule",
    "any_of",
    "canonicalize",
    "field_in",
    "field_truthy",
    "get_adapter",
    "list_adapters",
    "make_action_adapter",
    "make_event_adapter",
    "register_adapter",
]
