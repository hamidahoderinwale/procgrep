"""Compare canonical actions, native actions, and transitions in two corpora.

Counts are pooled within each corpus. Unlike ``jsd_matrix``, longer traces
therefore receive more weight. Every divergence uses the shared ``jsd``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import median
from typing import Any, TypeVar

from procgrep.jsd import jsd

_Token = TypeVar("_Token", str, tuple[str, str])


@dataclass(frozen=True)
class AgentSummary:
    """Trace counts, outcomes, lengths, and composition at the peak step."""

    name: str
    n_traces: int
    n_labeled: int
    n_resolved: int
    median_steps: float
    peak_fractions: tuple[float, ...]

    def pass_rate_text(self) -> str:
        if not self.n_labeled:
            return "n/a"
        return f"{self.n_resolved / self.n_labeled:.1%} ({self.n_labeled} labeled)"


@dataclass(frozen=True)
class ComparisonReport:
    """Shared numerical result for the Python API and CLI comparison."""

    a: AgentSummary
    b: AgentSummary
    canonical_jsd: float | None
    native_jsd: float | None
    bigram_jsd: float | None
    alphabet: tuple[str, ...]
    positional_curve: tuple[float, ...]
    top_a: tuple[tuple[str, float], ...]
    top_b: tuple[tuple[str, float], ...]
    base: float
    position_limit: int
    min_position_traces: int

    @property
    def peak_step(self) -> int | None:
        if not self.positional_curve:
            return None
        return max(range(len(self.positional_curve)), key=self.positional_curve.__getitem__)

    @property
    def peak_step_jsd(self) -> float | None:
        step = self.peak_step
        return None if step is None else self.positional_curve[step]

    def to_dict(self) -> dict[str, Any]:
        """Versioned JSON report; absent measurements remain null."""
        return {
            "schema_version": 2,
            "base": self.base,
            "aggregation": "pooled_counts",
            "smoothing": 0.0,
            "canonical_alphabet": list(self.alphabet),
            "position_limit": self.position_limit,
            "min_position_traces": self.min_position_traces,
            "agent_a": self.a.name,
            "agent_b": self.b.name,
            "n_a": self.a.n_traces,
            "n_b": self.b.n_traces,
            "canonical_jsd": _rounded(self.canonical_jsd),
            "native_jsd": _rounded(self.native_jsd),
            "bigram_jsd": _rounded(self.bigram_jsd),
            "peak_step": self.peak_step,
            "peak_step_jsd": _rounded(self.peak_step_jsd),
            "positional_curve": [round(v, 5) for v in self.positional_curve],
            "discriminative_bigrams": {
                self.a.name: [{"bigram": bg, "delta_p": round(d, 5)} for bg, d in self.top_a],
                self.b.name: [{"bigram": bg, "delta_p": round(d, 5)} for bg, d in self.top_b],
            },
        }

    def summary(self) -> str:
        """Render the CLI summary from the computed report."""
        lines = [
            f"\n{'=' * 64}",
            f"  {self.a.name}  ({self.a.n_traces} trajectories)",
            f"  {self.b.name}  ({self.b.n_traces} trajectories)",
            f"  JSD logarithm base: {self.base:g}; pooled counts",
            "=" * 64,
            f"\n  Canonical JSD : {_display(self.canonical_jsd)}  (atom-level composition)",
            f"  Native JSD    : {_display(self.native_jsd)}  (scaffold-specific tool usage)",
            f"\n  Pass rate  {self.a.name}: {self.a.pass_rate_text()}",
            f"  Pass rate  {self.b.name}: {self.b.pass_rate_text()}",
            f"\n  Median steps  {self.a.name}: {self.a.median_steps:.0f}",
            f"  Median steps  {self.b.name}: {self.b.median_steps:.0f}",
            f"\n  Bigram JSD    : {_display(self.bigram_jsd)}  (transition structure)",
        ]
        for agent, top in ((self.a, self.top_a), (self.b, self.top_b)):
            lines.append(f"\n  {agent.name} signature (over-represented transitions):")
            for bigram, delta in top:
                left, right = bigram.split("|", 1)
                lines.append(f"    {left:14s}→ {right:14s}  Δp={delta:+.4f}")
        if self.peak_step is None:
            lines.append("\n  Peak positional divergence: unavailable (insufficient traces)")
        else:
            lines.append(
                f"\n  Peak positional divergence: step {self.peak_step} "
                f"(JSD={_display(self.peak_step_jsd)})"
            )
            lines.append("  Step  " + "  ".join(f"{a[:6]:>6s}" for a in self.alphabet))
            for agent in (self.a, self.b):
                fractions = "  ".join(f"{v:>6.2f}" for v in agent.peak_fractions)
                lines.append(f"  {agent.name[:18]:>18s}  {fractions}")
        lines.append(f"\n{'=' * 64}\n")
        return "\n".join(lines)


def build_comparison(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    name_a: str = "A",
    name_b: str = "B",
    top_n: int = 5,
    base: float = 2.0,
    position_limit: int = 40,
    min_position_traces: int = 10,
) -> ComparisonReport:
    """Compare records containing ``atoms_canonical`` and optional native actions.

    ``atoms_native`` and ``resolved`` retain the legacy comparison input shape.
    A divergence is unavailable if either corpus has no relevant observations.
    Positions require ``min_position_traces`` observations on each side.
    """
    if not rows_a or not rows_b:
        empty = name_a if not rows_a else name_b
        raise ValueError(f"no trajectories in {empty}; both inputs must be non-empty")
    if name_a == name_b:
        raise ValueError("comparison names must be distinct")
    if not math.isfinite(base) or base <= 1:
        raise ValueError("base must be finite and greater than 1")
    if top_n < 0 or position_limit < 1 or min_position_traces < 1:
        raise ValueError(
            "top_n must be non-negative; position limits and minimums must be positive"
        )

    seqs_a = _sequences(rows_a, "atoms_canonical", required=True)
    seqs_b = _sequences(rows_b, "atoms_canonical", required=True)
    counts_a, counts_b = _counts(seqs_a), _counts(seqs_b)
    alphabet = tuple(sorted(counts_a.keys() | counts_b.keys()))
    native_a = _counts(_sequences(rows_a, "atoms_native"))
    native_b = _counts(_sequences(rows_b, "atoms_native"))
    bigrams_a, bigrams_b = _bigrams(seqs_a), _bigrams(seqs_b)
    curve = _position_curve(seqs_a, seqs_b, base, position_limit, min_position_traces)
    peak = max(range(len(curve)), key=curve.__getitem__) if curve else None
    top_a, top_b = _top_bigrams(bigrams_a, bigrams_b, top_n)
    return ComparisonReport(
        a=_agent_summary(name_a, rows_a, seqs_a, alphabet, peak),
        b=_agent_summary(name_b, rows_b, seqs_b, alphabet, peak),
        canonical_jsd=_distance(counts_a, counts_b, base),
        native_jsd=_distance(native_a, native_b, base),
        bigram_jsd=_distance(bigrams_a, bigrams_b, base),
        alphabet=alphabet,
        positional_curve=curve,
        top_a=top_a,
        top_b=top_b,
        base=base,
        position_limit=position_limit,
        min_position_traces=min_position_traces,
    )


def _sequences(
    rows: Sequence[Mapping[str, Any]], field: str, *, required: bool = False
) -> list[list[str]]:
    sequences = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("comparison records must be mappings")
        if required and field not in row:
            raise ValueError(f"comparison records require {field!r}")
        value = row.get(field, [])
        if not isinstance(value, (list, tuple)) or any(not isinstance(a, str) for a in value):
            raise ValueError(f"{field} must be a sequence of action names")
        sequences.append(list(value))
    return sequences


def _counts(sequences: Sequence[Sequence[str]]) -> Counter[str]:
    return Counter(atom for seq in sequences for atom in seq)


def _bigrams(sequences: Sequence[Sequence[str]]) -> Counter[tuple[str, str]]:
    return Counter(pair for seq in sequences for pair in pairwise(seq))


def _distance(a: Counter[_Token], b: Counter[_Token], base: float) -> float | None:
    if not a or not b:
        return None
    alphabet = sorted(a.keys() | b.keys())
    return jsd([a[token] for token in alphabet], [b[token] for token in alphabet], base=base)


def _position_curve(
    a: Sequence[Sequence[str]],
    b: Sequence[Sequence[str]],
    base: float,
    limit: int,
    minimum: int,
) -> tuple[float, ...]:
    curve = []
    for k in range(limit):
        ca = Counter(seq[k] for seq in a if k < len(seq))
        cb = Counter(seq[k] for seq in b if k < len(seq))
        if min(sum(ca.values()), sum(cb.values())) < minimum:
            break
        distance = _distance(ca, cb, base)
        if distance is not None:
            curve.append(distance)
    return tuple(curve)


def _top_bigrams(
    a: Counter[tuple[str, str]], b: Counter[tuple[str, str]], n: int
) -> tuple[tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    if not a or not b:
        return (), ()
    na, nb = sum(a.values()), sum(b.values())
    differences = [(a[pair] / na - b[pair] / nb, pair) for pair in a.keys() | b.keys()]
    top_a = tuple(
        ("|".join(pair), delta) for delta, pair in sorted(differences, reverse=True) if delta > 0
    )[:n]
    top_b = tuple(("|".join(pair), -delta) for delta, pair in sorted(differences) if delta < 0)[:n]
    return top_a, top_b


def _agent_summary(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    sequences: Sequence[Sequence[str]],
    alphabet: tuple[str, ...],
    peak: int | None,
) -> AgentSummary:
    labeled = [r for r in rows if r.get("resolved") is not None]
    counts = (
        Counter(seq[peak] for seq in sequences if peak < len(seq))
        if peak is not None
        else Counter()
    )
    total = sum(counts.values())
    return AgentSummary(
        name=name,
        n_traces=len(rows),
        n_labeled=len(labeled),
        n_resolved=sum(bool(r["resolved"]) for r in labeled),
        median_steps=float(median(len(seq) for seq in sequences)),
        peak_fractions=tuple(counts[atom] / total for atom in alphabet) if total else (),
    )


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 5)


def _display(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


__all__ = ["AgentSummary", "ComparisonReport", "build_comparison"]
