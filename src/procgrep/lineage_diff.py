"""Structured procedural-level diff between a parent and a child model.

`lineage_diff` characterizes what a documented training procedure
(distillation, SFT, RLHF, instruction tuning, version step) did at
the procedural level. Given canonical trajectories from the parent
and the child evaluated on a shared task suite, it produces a
structured :class:`LineageDiff` object with one :class:`AxisResult`
per requested measurement axis.

This module composes existing `procgrep` primitives (atom frequencies,
per-trajectory entropy) into an aggregation suitable for verifying
whether a training-method paper's preservation claims hold up in
trajectories. The diff *object* is the contribution; individual diffs
are data points, and a collection of diffs across many lineage steps
becomes a reference catalog of "what training procedures do
procedurally".

The MVP ships three axes:

* ``"vocabulary"`` — Jaccard similarity of the *set* of canonical
  atoms produced. Surfaces vocabulary collapse (procedures lost in
  the child) and vocabulary growth (procedures the parent never
  emitted).
* ``"entropy"`` — Mean per-trajectory Shannon entropy on each side.
  The shift (parent minus child) is positive when the child
  concentrates on a smaller set of procedures (mode-collapse signature).
* ``"outcome_quadrant"`` — Requires an outcome label in trace
  metadata (e.g., ``resolved: bool``). Decomposes the diff into
  pass / fail strata and reports per-stratum vocabulary preservation.
  Lets a reader separate "the child preserved successful procedures"
  from "the child preserved failure modes".

Additional axes (``"conditional"``, ``"recovery"``, ``"failures"``,
``"ood"``, ``"phase"``) are designed but not implemented in the MVP;
they will be added as concrete audits demand them. Each new axis is a
separate function with the same :class:`AxisResult` return type.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from procgrep.types import Atom, Trace

DEFAULT_AXES: tuple[str, ...] = ("vocabulary", "entropy")
"""Axes computed when no ``along=`` argument is passed to :func:`lineage_diff`."""


@dataclass(frozen=True)
class AxisResult:
    """One axis of a lineage diff.

    Attributes:
        axis: Short name of the axis, matching the requested key
            (``"vocabulary"``, ``"entropy"``, ``"outcome_quadrant"``).
        summary_value: A single scalar capturing the axis's headline
            measurement (Jaccard similarity, entropy shift, etc.).
            Interpretation depends on the axis; see the module
            docstring.
        detail: Structured per-axis detail (sets of preserved /
            lost / gained items, mean and median statistics,
            interpretation notes). All values are JSON-serializable.
        alphabet: Atom alphabet under which this axis was computed.
            Defaults to ``"canonical"`` (the shared canonical alphabet
            used by built-in adapters). Other common values include
            ``"native"`` (the scaffold's own action vocabulary). Users
            may pass any string; the field is informational, used by
            readers of the diff to interpret results correctly when
            multiple alphabets are reported in one LineageDiff.
    """

    axis: str
    summary_value: float
    detail: Mapping[str, object]
    alphabet: str = "canonical"


@dataclass(frozen=True)
class LineageDiff:
    """Structured diff between two procedural distributions.

    Returned by :func:`lineage_diff`. Composes existing `procgrep`
    primitives into a research artifact suitable for paper sections,
    automated audit reports, or contributions to a community delta
    catalog.

    Attributes:
        parent_label: Identifier for the parent in output (model name,
            checkpoint id, etc.).
        child_label: Identifier for the child in output.
        n_parent: Number of parent-side trajectories analyzed.
        n_child: Number of child-side trajectories analyzed.
        axes: Ordered tuple of :class:`AxisResult`, one per requested
            axis, in the same order as the ``along=`` argument.
    """

    parent_label: str
    child_label: str
    n_parent: int
    n_child: int
    axes: tuple[AxisResult, ...]

    def summary(self) -> str:
        """Human-readable one-line-per-axis summary."""
        lines = [
            f"LineageDiff: {self.parent_label} -> {self.child_label}",
            f"  n_parent={self.n_parent}, n_child={self.n_child}",
        ]
        for axis in self.axes:
            lines.append(f"  {axis.axis}: {axis.summary_value:.4f}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Markdown audit report, suitable for paper sections or PR comments."""
        lines = [
            f"# Lineage diff: `{self.parent_label}` -> `{self.child_label}`",
            "",
            f"- Parent trajectories analyzed: {self.n_parent}",
            f"- Child trajectories analyzed: {self.n_child}",
            "",
            "## Axes",
            "",
        ]
        for axis in self.axes:
            lines.append(f"### {axis.axis}")
            lines.append("")
            lines.append(f"Summary: **{axis.summary_value:.4f}**")
            lines.append("")
            for key, value in axis.detail.items():
                if isinstance(value, list):
                    preview = ", ".join(repr(x) for x in value[:10])
                    suffix = f" (+{len(value) - 10} more)" if len(value) > 10 else ""
                    lines.append(f"- **{key}** ({len(value)}): {preview}{suffix}")
                elif isinstance(value, float):
                    lines.append(f"- **{key}**: {value:.4f}")
                else:
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        return "\n".join(lines)

    def to_records(self) -> dict[str, object]:
        """JSON-friendly dict representation suitable for catalog entries."""
        return {
            "parent_label": self.parent_label,
            "child_label": self.child_label,
            "n_parent": self.n_parent,
            "n_child": self.n_child,
            "axes": [
                {
                    "axis": axis.axis,
                    "summary_value": axis.summary_value,
                    "detail": dict(axis.detail),
                    "alphabet": axis.alphabet,
                }
                for axis in self.axes
            ],
        }


def lineage_diff(
    parent: Sequence[Trace],
    child: Sequence[Trace],
    *,
    parent_label: str = "parent",
    child_label: str = "child",
    along: Sequence[str] = DEFAULT_AXES,
    outcome_field: str | None = None,
    alphabet: str | Sequence[str] = "canonical",
    canonical_projection: Callable[[Atom], Atom] | None = None,
) -> LineageDiff:
    """Compute a structured procedural-level diff.

    Composes `procgrep` primitives into a structured-diff object
    characterizing how the child's procedural distribution differs
    from the parent's along the requested axes.

    Hierarchical multi-resolution: pass ``alphabet=["canonical",
    "native"]`` to run each axis under both atom alphabets in a
    single diff. The resulting LineageDiff carries one AxisResult
    per (axis, alphabet) pair, each tagged with which alphabet it
    came from. Useful when you want both cross-comparable canonical
    results (for catalog aggregation) AND scaffold-native richness
    (for within-scaffold depth) in a single audit.

    Args:
        parent: Parent-side canonical traces (e.g., from the base model
            on a shared task suite).
        child: Child-side canonical traces (e.g., from the post-trained
            model on the same task suite).
        parent_label: Identifier used in the output (model name,
            checkpoint id, etc.).
        child_label: Identifier used in the output.
        along: Names of axes to compute. Defaults to
            ``("vocabulary", "entropy")``. The ``"outcome_quadrant"``
            axis additionally requires ``outcome_field`` to identify
            the per-trace metadata field carrying the outcome label.
        outcome_field: Name of the metadata field carrying the binary
            outcome label for each trace (e.g., ``"resolved"``).
            Required if ``"outcome_quadrant"`` is among the requested
            axes; ignored otherwise.
        alphabet: Atom alphabet(s) to compute axes under. A single
            string (default ``"canonical"``) runs each requested axis
            once, tagged with that alphabet. A sequence of strings
            (e.g., ``["canonical", "native"]``) runs each axis once
            per alphabet; the resulting LineageDiff.axes contains all
            results in nested order (alphabet outer, axis inner),
            with each AxisResult.alphabet field recording the source.
            Alphabet names are informational labels; the actual
            projection (if any) is supplied via canonical_projection.
        canonical_projection: Optional callable mapping each native
            atom to its canonical equivalent. Applied to all traces
            when computing axes under the ``"canonical"`` alphabet.
            Leave as ``None`` if your traces already emit canonical
            atoms (which is the default for the built-in adapters).
            When projection is None, the ``"canonical"`` mode is a
            no-op pass-through.

    Returns:
        A :class:`LineageDiff` with one :class:`AxisResult` per
        requested (axis, alphabet) pair.

    Raises:
        ValueError: If a requested axis name is not recognized, or if
            ``"outcome_quadrant"`` is requested without
            ``outcome_field``.
    """
    parent_list = list(parent)
    child_list = list(child)

    alphabets: list[str] = [alphabet] if isinstance(alphabet, str) else list(alphabet)

    all_axes: list[AxisResult] = []
    for alpha in alphabets:
        # Apply the canonical projection only when computing under
        # the "canonical" alphabet. Other named alphabets are
        # treated as the trace's native form and run on as-emitted
        # atoms. Without a projection, "canonical" is a no-op label.
        if alpha == "canonical" and canonical_projection is not None:
            p_traces = _project_traces(parent_list, canonical_projection)
            c_traces = _project_traces(child_list, canonical_projection)
        else:
            p_traces, c_traces = parent_list, child_list

        for axis_name in along:
            result = _compute_axis(axis_name, p_traces, c_traces, outcome_field)
            all_axes.append(replace(result, alphabet=alpha))

    return LineageDiff(
        parent_label=parent_label,
        child_label=child_label,
        n_parent=len(parent_list),
        n_child=len(child_list),
        axes=tuple(all_axes),
    )


def _compute_axis(
    axis_name: str,
    parent: list[Trace],
    child: list[Trace],
    outcome_field: str | None,
) -> AxisResult:
    """Dispatch a single axis computation by name.

    Each axis is a separate function with the same signature shape;
    the dispatch is centralized here so the alphabet-loop in
    :func:`lineage_diff` stays compact.
    """
    if axis_name == "vocabulary":
        return _diff_vocabulary(parent, child)
    if axis_name == "entropy":
        return _diff_entropy(parent, child)
    if axis_name == "outcome_quadrant":
        if outcome_field is None:
            raise ValueError(
                "axis 'outcome_quadrant' requires outcome_field= "
                "(name of the metadata field carrying the binary outcome label)"
            )
        return _diff_outcome_quadrant(parent, child, outcome_field)
    raise ValueError(
        f"unknown axis: {axis_name!r}; available: " "'vocabulary', 'entropy', 'outcome_quadrant'"
    )


def _project_traces(
    traces: list[Trace],
    projection: Callable[[Atom], Atom],
) -> list[Trace]:
    """Return new Trace objects with atoms projected through ``projection``.

    Other fields (trace_id, agent, group, metadata) are preserved.
    Used by :func:`lineage_diff` to apply a canonical_projection when
    requested without mutating the caller's input.
    """
    return [
        Trace(
            trace_id=t.trace_id,
            agent=t.agent,
            atoms=[projection(a) for a in t.atoms],
            group=t.group,
            metadata=t.metadata,
        )
        for t in traces
    ]


def _diff_vocabulary(parent: list[Trace], child: list[Trace]) -> AxisResult:
    """Compute atom-vocabulary preservation as Jaccard similarity.

    Reports the set of atoms each side emits at least once, plus
    Jaccard similarity, plus the per-side disjoint sets (lost /
    gained vocabulary).
    """
    parent_atoms = _collect_atoms(parent)
    child_atoms = _collect_atoms(child)
    intersection = parent_atoms & child_atoms
    union = parent_atoms | child_atoms
    jaccard = len(intersection) / len(union) if union else 1.0
    return AxisResult(
        axis="vocabulary",
        summary_value=jaccard,
        detail={
            "jaccard": jaccard,
            "shared": sorted(intersection),
            "parent_only": sorted(parent_atoms - child_atoms),
            "child_only": sorted(child_atoms - parent_atoms),
            "parent_vocab_size": len(parent_atoms),
            "child_vocab_size": len(child_atoms),
        },
    )


def _diff_entropy(parent: list[Trace], child: list[Trace]) -> AxisResult:
    """Compare per-trajectory atom-distribution entropy across sides.

    Computes mean and median per-trajectory Shannon entropy for each
    side. The headline summary is ``parent_mean - child_mean``;
    positive values indicate the child is more concentrated on a
    smaller set of procedures (a mode-collapse signature).
    """
    parent_entropies = [_atom_entropy(trace.atoms) for trace in parent]
    child_entropies = [_atom_entropy(trace.atoms) for trace in child]
    parent_mean = _mean(parent_entropies)
    child_mean = _mean(child_entropies)
    return AxisResult(
        axis="entropy",
        summary_value=parent_mean - child_mean,
        detail={
            "parent_mean": parent_mean,
            "parent_median": _median(parent_entropies),
            "child_mean": child_mean,
            "child_median": _median(child_entropies),
            "shift": parent_mean - child_mean,
            "interpretation": (
                "positive shift = child more concentrated (mode collapse); "
                "negative = child more diverse"
            ),
        },
    )


def _diff_outcome_quadrant(
    parent: list[Trace],
    child: list[Trace],
    outcome_field: str,
) -> AxisResult:
    """Decompose the diff by outcome (pass / fail) for each side.

    Partitions each side's traces by the truthiness of
    ``trace.metadata[outcome_field]``, then reports per-stratum
    vocabulary preservation. Lets a reader separate "the child
    preserved procedures on tasks that pass" from "the child preserved
    failure-mode procedures".

    The headline summary is pass-stratum Jaccard; the detail includes
    fail-stratum Jaccard alongside per-stratum trajectory counts.
    """
    parent_pass, parent_fail = _split_by_outcome(parent, outcome_field)
    child_pass, child_fail = _split_by_outcome(child, outcome_field)
    pass_jaccard = _jaccard(_collect_atoms(parent_pass), _collect_atoms(child_pass))
    fail_jaccard = _jaccard(_collect_atoms(parent_fail), _collect_atoms(child_fail))
    return AxisResult(
        axis="outcome_quadrant",
        summary_value=pass_jaccard,
        detail={
            "outcome_field": outcome_field,
            "parent_pass_n": len(parent_pass),
            "parent_fail_n": len(parent_fail),
            "child_pass_n": len(child_pass),
            "child_fail_n": len(child_fail),
            "pass_vocab_jaccard": pass_jaccard,
            "fail_vocab_jaccard": fail_jaccard,
            "interpretation": (
                "pass_vocab_jaccard near 1 = faithful preservation when succeeding; "
                "fail_vocab_jaccard near 1 = preserved failure modes; "
                "large gap between the two = outcome-dependent procedural drift"
            ),
        },
    )


# Helpers --------------------------------------------------------------------


def _collect_atoms(traces: list[Trace]) -> set[Atom]:
    """Return the set of distinct atoms appearing across the traces."""
    out: set[Atom] = set()
    for trace in traces:
        out.update(trace.atoms)
    return out


def _atom_entropy(atoms: Sequence[Atom]) -> float:
    """Shannon entropy of one trajectory's atom-frequency distribution, in nats.

    Returns 0.0 for an empty trajectory or one with all-identical
    atoms. The maximum is ``log(k)`` where ``k`` is the number of
    distinct atoms in the trajectory.
    """
    if not atoms:
        return 0.0
    counts = Counter(atoms)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values() if count > 0)


def _split_by_outcome(
    traces: list[Trace],
    outcome_field: str,
) -> tuple[list[Trace], list[Trace]]:
    """Partition traces into (pass, fail) lists by the truthiness of a metadata field."""
    pass_: list[Trace] = []
    fail: list[Trace] = []
    for trace in traces:
        if bool(trace.metadata.get(outcome_field)):
            pass_.append(trace)
        else:
            fail.append(trace)
    return pass_, fail


def _jaccard(a: set[Atom], b: set[Atom]) -> float:
    """Jaccard similarity of two atom sets, defined as 1.0 when both are empty."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean; returns 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    """Median; returns 0.0 for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# Lightweight type alias for any axis-computing function. Exposed so
# external packages can register additional axes in the future without
# importing from this module's private namespace.
AxisFn = Callable[[list[Trace], list[Trace]], AxisResult]


__all__ = [
    "DEFAULT_AXES",
    "AxisFn",
    "AxisResult",
    "LineageDiff",
    "lineage_diff",
]
