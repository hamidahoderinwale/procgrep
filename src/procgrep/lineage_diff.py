"""Structured procedural-level diff between a parent and a child model.

Given canonical trajectories from a parent and child on a shared task
suite, :func:`lineage_diff` returns a :class:`LineageDiff` with one
:class:`AxisResult` per requested axis. Each diff is a data point; a
collection across many lineage steps becomes a catalog of what
training procedures do at the procedural level.

Axes:

* ``"vocabulary"`` -- Jaccard similarity over the set of atoms emitted.
* ``"entropy"`` -- Mean per-trajectory entropy on each side; shift
  (parent minus child) is positive under child mode-collapse.
* ``"outcome_quadrant"`` -- Requires an outcome label in metadata.
  Reports vocabulary preservation per pass / fail stratum.
* ``"conditional"`` -- Markov-conditional JSD between parent and
  child next-atom distributions given the previous
  ``conditional_context_k`` atoms. Catches sequence-structure drift
  that marginal axes miss.

Additional axes (``"recovery"``, ``"failures"``, ``"ood"``,
``"phase"``) are designed but unimplemented; add them as concrete
audits require.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from procgrep.jsd import jsd as _jsd
from procgrep.types import Atom, Trace

DEFAULT_AXES: tuple[str, ...] = ("vocabulary", "entropy")
"""Default axes when no ``along=`` is passed to :func:`lineage_diff`."""


@dataclass(frozen=True)
class AxisResult:
    """One axis of a lineage diff.

    Attributes:
        axis: Short axis name (``"vocabulary"``, ``"entropy"``,
            ``"outcome_quadrant"``, ``"conditional"``).
        summary_value: Headline scalar; interpretation depends on the
            axis (see module docstring).
        detail: Per-axis structured detail. JSON-serializable.
        alphabet: Informational label identifying which atom alphabet
            this axis was computed under (``"canonical"`` or
            ``"native"`` are conventional; any string is allowed).
    """

    axis: str
    summary_value: float
    detail: Mapping[str, object]
    alphabet: str = "canonical"


@dataclass(frozen=True)
class LineageDiff:
    """Structured diff between two procedural distributions.

    Returned by :func:`lineage_diff`. Suitable for paper sections,
    automated audit reports, and catalog entries.

    Attributes:
        axes: One :class:`AxisResult` per requested axis, in the same
            order as the ``along=`` argument (and repeated per
            alphabet when ``alphabet=`` is a sequence).
    """

    parent_label: str
    child_label: str
    n_parent: int
    n_child: int
    axes: tuple[AxisResult, ...]

    def summary(self) -> str:
        """One-line-per-axis text summary."""
        lines = [
            f"LineageDiff: {self.parent_label} -> {self.child_label}",
            f"  n_parent={self.n_parent}, n_child={self.n_child}",
        ]
        for axis in self.axes:
            lines.append(f"  {axis.axis}: {axis.summary_value:.4f}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Markdown audit report for paper sections or PR comments."""
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
        """JSON-friendly dict for catalog entries."""
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
    conditional_context_k: int = 1,
) -> LineageDiff:
    """Compute a structured procedural-level diff.

    Pass ``alphabet=["canonical", "native"]`` to run each axis under
    both alphabets in one diff; the result's ``axes`` is ordered with
    alphabet outer and axis inner, each tagged on
    ``AxisResult.alphabet``.

    Args:
        along: Axes to compute. ``"outcome_quadrant"`` also needs
            ``outcome_field``.
        outcome_field: Metadata key holding the binary outcome
            (e.g. ``"resolved"``).
        alphabet: One label or a sequence of labels. ``"canonical"``
            applies ``canonical_projection`` when supplied; other
            labels run on as-emitted atoms.
        canonical_projection: Native-to-canonical atom map. ``None``
            (default) makes the ``"canonical"`` pass a no-op.
        conditional_context_k: Prefix length for the ``"conditional"``
            axis. Larger k captures longer-range structure but yields
            sparser distributions.

    Raises:
        ValueError: Unknown axis name, or ``"outcome_quadrant"``
            requested without ``outcome_field``.
    """
    parent_list = list(parent)
    child_list = list(child)

    alphabets: list[str] = [alphabet] if isinstance(alphabet, str) else list(alphabet)

    all_axes: list[AxisResult] = []
    for alpha in alphabets:
        # Only the "canonical" pass applies the projection; other
        # labels run on as-emitted atoms.
        if alpha == "canonical" and canonical_projection is not None:
            p_traces = _project_traces(parent_list, canonical_projection)
            c_traces = _project_traces(child_list, canonical_projection)
        else:
            p_traces, c_traces = parent_list, child_list

        for axis_name in along:
            result = _compute_axis(
                axis_name,
                p_traces,
                c_traces,
                outcome_field,
                conditional_context_k=conditional_context_k,
            )
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
    *,
    conditional_context_k: int = 1,
) -> AxisResult:
    """Dispatch one axis computation by name."""
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
    if axis_name == "conditional":
        return _diff_conditional(parent, child, context_k=conditional_context_k)
    raise ValueError(
        f"unknown axis: {axis_name!r}; available: "
        "'vocabulary', 'entropy', 'outcome_quadrant', 'conditional'"
    )


def _project_traces(
    traces: list[Trace],
    projection: Callable[[Atom], Atom],
) -> list[Trace]:
    """New Traces with atoms projected through ``projection``.

    Other fields are preserved. Caller input is not mutated.
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
    """Atom-vocabulary preservation as Jaccard similarity.

    Reports each side's atom set, the Jaccard, and the disjoint
    parent-only / child-only sets.
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
    """Per-trajectory atom-entropy comparison.

    Summary is ``parent_mean - child_mean``. Positive => child is
    more concentrated (mode collapse).
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


def _diff_conditional(
    parent: list[Trace],
    child: list[Trace],
    *,
    context_k: int = 1,
) -> AxisResult:
    """Markov-conditional JSD between parent and child next-atom choices.

    For each prefix of length ``context_k`` shared by both corpora,
    computes the empirical next-atom distribution on each side and
    their JSD. Summary is the parent-frequency-weighted mean across
    shared prefixes (rare prefixes get small weight).

    Catches sequence-structure drift that marginal axes miss: a child
    that always picks ``run_test`` after ``edit`` where the parent
    picked another ``edit`` shows high JSD on prefix ``("edit",)``
    even at identical marginal counts. Inspect
    ``top_divergent_prefixes`` for tail patterns the weighted mean
    suppresses.
    """
    if context_k < 1:
        raise ValueError(f"context_k must be >= 1, got {context_k}")
    parent_cond = _collect_conditionals(parent, context_k)
    child_cond = _collect_conditionals(child, context_k)
    shared = set(parent_cond) & set(child_cond)
    parent_only = set(parent_cond) - set(child_cond)
    child_only = set(child_cond) - set(parent_cond)

    if not shared:
        return AxisResult(
            axis="conditional",
            summary_value=0.0,
            detail={
                "context_k": context_k,
                "shared_prefixes": 0,
                "parent_only_prefixes": len(parent_only),
                "child_only_prefixes": len(child_only),
                "interpretation": (
                    "no prefixes shared between parent and child at context_k; "
                    "weighted-mean JSD is undefined (returned as 0.0). "
                    "Inspect parent_only / child_only counts."
                ),
            },
        )

    parent_total = sum(sum(c.values()) for c in parent_cond.values())
    per_prefix: list[tuple[tuple[Atom, ...], float, int]] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for prefix in shared:
        p_counts = parent_cond[prefix]
        c_counts = child_cond[prefix]
        keys = sorted(set(p_counts) | set(c_counts))
        p_vec = [p_counts[k] for k in keys]
        c_vec = [c_counts[k] for k in keys]
        prefix_jsd = _jsd(p_vec, c_vec)
        prefix_freq = sum(p_counts.values())
        weight = prefix_freq / parent_total if parent_total > 0 else 0.0
        per_prefix.append((prefix, prefix_jsd, prefix_freq))
        weighted_sum += prefix_jsd * weight
        total_weight += weight

    mean_jsd = weighted_sum / total_weight if total_weight > 0 else 0.0
    top_divergent = sorted(per_prefix, key=lambda t: -t[1])[:10]

    return AxisResult(
        axis="conditional",
        summary_value=mean_jsd,
        detail={
            "context_k": context_k,
            "shared_prefixes": len(shared),
            "parent_only_prefixes": len(parent_only),
            "child_only_prefixes": len(child_only),
            "top_divergent_prefixes": [
                {"prefix": list(prefix), "jsd": jsd_val, "parent_freq": freq}
                for prefix, jsd_val, freq in top_divergent
            ],
            "interpretation": (
                "Mean JSD between P(next | prefix) for parent and child on "
                "shared prefixes, weighted by parent prefix frequency. "
                "Higher = more conditional structure drift. 0 = identical "
                "conditional distributions on every shared prefix."
            ),
        },
    )


def _collect_conditionals(
    traces: list[Trace],
    context_k: int,
) -> dict[tuple[Atom, ...], Counter[Atom]]:
    """Markov-conditional table from a corpus.

    Maps each ``context_k``-atom prefix to a Counter over the atom
    that followed it. Traces shorter than ``context_k + 1`` are
    skipped.
    """
    conds: dict[tuple[Atom, ...], Counter[Atom]] = defaultdict(Counter)
    for trace in traces:
        atoms = trace.atoms
        if len(atoms) <= context_k:
            continue
        for i in range(context_k, len(atoms)):
            prefix = tuple(atoms[i - context_k : i])
            conds[prefix][atoms[i]] += 1
    return conds


def _diff_outcome_quadrant(
    parent: list[Trace],
    child: list[Trace],
    outcome_field: str,
) -> AxisResult:
    """Decompose the diff by outcome (pass / fail).

    Partitions each side on the truthiness of
    ``trace.metadata[outcome_field]`` and reports per-stratum
    vocabulary preservation. Summary is the pass-stratum Jaccard;
    detail also carries the fail-stratum Jaccard and per-stratum
    counts.
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


def _collect_atoms(traces: list[Trace]) -> set[Atom]:
    """Distinct atoms appearing across the traces."""
    out: set[Atom] = set()
    for trace in traces:
        out.update(trace.atoms)
    return out


def _atom_entropy(atoms: Sequence[Atom]) -> float:
    """Shannon entropy of one trajectory's atom distribution, in nats.

    Returns 0.0 on empty or single-atom trajectories. Max is
    ``log(k)`` for k distinct atoms.
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
    """Partition traces by truthiness of ``metadata[outcome_field]``."""
    pass_: list[Trace] = []
    fail: list[Trace] = []
    for trace in traces:
        if bool(trace.metadata.get(outcome_field)):
            pass_.append(trace)
        else:
            fail.append(trace)
    return pass_, fail


def _jaccard(a: set[Atom], b: set[Atom]) -> float:
    """Jaccard similarity; 1.0 when both sets are empty."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 on empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    """Median; 0.0 on empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


# Exposed so external packages can register additional axes without
# touching this module's private namespace.
AxisFn = Callable[[list[Trace], list[Trace]], AxisResult]


__all__ = [
    "DEFAULT_AXES",
    "AxisFn",
    "AxisResult",
    "LineageDiff",
    "lineage_diff",
]
