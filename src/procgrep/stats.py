"""Group-level descriptive and discriminative statistics.

Convenience layer over `canonicalize` / `fit_bpe` / `encode` / `jsd`:

* `atom_frequencies_per_group` -- dominant raw atoms per group.
* `effective_vocab_size_per_group` -- group-mean diversity as an
  equivalent uniform-procedure count.
* `entropies_per_group` -- per-trajectory entropy summary per group.
* `discriminative_procedures` -- procedures distinguishing two groups,
  ranked by log-odds or by JSD contribution.

All functions are pure over `Trace`, `Fingerprint`, and
`ProcedureVocabulary`.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from procgrep.bpe import ProcedureVocabulary
from procgrep.encode import Fingerprint
from procgrep.types import Atom, Trace

GroupBy = Literal["group", "agent"]
Ranking = Literal["log_odds", "jsd_contribution"]


@dataclass(frozen=True)
class GroupAtomFrequencies:
    """Top-K most-frequent raw atoms in one group.

    Attributes:
        top: ``(atom, count, per_trajectory_rate)`` entries, up to
            length ``k``.
    """

    group: str
    n_trajectories: int
    n_atoms_total: int
    top: tuple[tuple[Atom, int, float], ...]


@dataclass(frozen=True)
class GroupEntropyStats:
    """Per-trajectory Shannon entropy summary for one group.

    All entropy values are in nats. ``q1`` / ``q3`` are the 25th /
    75th percentiles.
    """

    group: str
    n: int
    median: float
    q1: float
    q3: float
    min: float
    max: float


@dataclass(frozen=True)
class DiscriminativeProcedure:
    """One procedure that distinguishes two groups.

    Attributes:
        procedure: Token (atom or BPE-merged procedure).
        p_a, p_b: Group-mean probability of the procedure in each group.
        log_odds: ``log((p_a + eps) / (p_b + eps))``. Positive favors A,
            negative favors B.
        jsd_contribution: Non-negative contribution to the JSD between
            the two group means.
    """

    procedure: str
    p_a: float
    p_b: float
    log_odds: float
    jsd_contribution: float


def atom_frequencies_per_group(
    traces: Iterable[Trace],
    *,
    k: int = 20,
    group_by: GroupBy = "group",
) -> dict[str, GroupAtomFrequencies]:
    """Top-K most-frequent raw atoms per group.

    Operates on `Trace.atoms`, not on BPE procedures; for
    procedure-level frequencies use fingerprint counts directly.

    Args:
        group_by: ``"group"`` uses `Trace.grouping()`; ``"agent"``
            uses `Trace.agent`.
    """
    by_group_counts: dict[str, Counter[Atom]] = defaultdict(Counter)
    by_group_n: Counter[str] = Counter()
    for trace in traces:
        key = trace.agent if group_by == "agent" else trace.grouping()
        by_group_counts[key].update(trace.atoms)
        by_group_n[key] += 1
    out: dict[str, GroupAtomFrequencies] = {}
    for group, counts in by_group_counts.items():
        n_traj = by_group_n[group]
        n_total = int(sum(counts.values()))
        top = tuple(
            (atom, int(count), float(count) / max(n_traj, 1))
            for atom, count in counts.most_common(k)
        )
        out[group] = GroupAtomFrequencies(
            group=group,
            n_trajectories=n_traj,
            n_atoms_total=n_total,
            top=top,
        )
    return out


def effective_vocab_size_per_group(
    fingerprints: Iterable[Fingerprint],
    *,
    group_by: GroupBy = "group",
) -> dict[str, float]:
    """Effective vocabulary size per group.

    ``exp(Shannon entropy of the group-mean procedure distribution)``.
    A group with effective vocab ``N`` is as diverse as a uniform
    distribution over ``N`` procedures (equivalently, the perplexity
    of the group mean under itself). Always ``>= 1.0``.
    """
    by_group: dict[str, list[npt.NDArray[np.float64]]] = defaultdict(list)
    for fp in fingerprints:
        key = fp.agent if group_by == "agent" else fp.group
        by_group[key].append(fp.distribution())
    out: dict[str, float] = {}
    for group, dists in by_group.items():
        stacked = np.stack(dists, axis=0)
        mean = stacked.mean(axis=0)
        total = float(mean.sum())
        if total <= 0.0:
            out[group] = 1.0
            continue
        mean = mean / total
        positive = mean[mean > 0]
        entropy = float(-np.sum(positive * np.log(positive)))
        out[group] = float(math.exp(entropy))
    return out


def entropies_per_group(
    fingerprints: Iterable[Fingerprint],
    *,
    group_by: GroupBy = "group",
) -> dict[str, GroupEntropyStats]:
    """Per-trajectory Shannon entropy summarized by group.

    Each fingerprint's entropy via `Fingerprint.entropy()`, summarized
    per group with median, IQR, and range. Surfaces groups whose
    trajectories are monolithic (low entropy) vs. diverse (high).
    """
    by_group: dict[str, list[float]] = defaultdict(list)
    for fp in fingerprints:
        key = fp.agent if group_by == "agent" else fp.group
        by_group[key].append(fp.entropy())
    out: dict[str, GroupEntropyStats] = {}
    for group, values in by_group.items():
        arr = np.asarray(values, dtype=np.float64)
        out[group] = GroupEntropyStats(
            group=group,
            n=len(values),
            median=float(np.median(arr)),
            q1=float(np.percentile(arr, 25)),
            q3=float(np.percentile(arr, 75)),
            min=float(arr.min()),
            max=float(arr.max()),
        )
    return out


def discriminative_procedures(
    fingerprints: Iterable[Fingerprint],
    vocab: ProcedureVocabulary,
    *,
    group_a: str,
    group_b: str,
    k: int = 10,
    ranking: Ranking = "log_odds",
    epsilon: float = 1e-6,
    group_by: GroupBy = "group",
) -> list[DiscriminativeProcedure]:
    """Top-K procedures distinguishing ``group_a`` from ``group_b``.

    Ranks every procedure by absolute log-odds or by JSD contribution
    between the two group-mean distributions. Other groups in
    ``fingerprints`` are ignored. ``vocab`` must match the encoding.

    Args:
        ranking: ``"log_odds"`` (signed A-vs-B ratio, sorted by
            magnitude) or ``"jsd_contribution"`` (non-negative).
        epsilon: Log-odds smoothing.

    Raises:
        ValueError: If vocab size disagrees with fingerprint dim, or
            if ``ranking`` is unrecognized.
    """
    fps = list(fingerprints)
    mean_a = _group_mean(fps, group_a, group_by)
    mean_b = _group_mean(fps, group_b, group_by)
    tokens = vocab.tokens()
    if len(tokens) != mean_a.shape[0]:
        raise ValueError(
            f"vocab token count ({len(tokens)}) does not match fingerprint "
            f"dimension ({mean_a.shape[0]}); fingerprints must be encoded "
            "under the supplied vocabulary"
        )
    rows: list[DiscriminativeProcedure] = []
    for i, token in enumerate(tokens):
        pa = max(float(mean_a[i]), 0.0)
        pb = max(float(mean_b[i]), 0.0)
        log_odds = math.log((pa + epsilon) / (pb + epsilon))
        m = 0.5 * (pa + pb)
        jsd_c = 0.0
        if pa > 0 and m > 0:
            jsd_c += 0.5 * pa * math.log(pa / m)
        if pb > 0 and m > 0:
            jsd_c += 0.5 * pb * math.log(pb / m)
        rows.append(
            DiscriminativeProcedure(
                procedure=token,
                p_a=pa,
                p_b=pb,
                log_odds=log_odds,
                jsd_contribution=float(jsd_c),
            )
        )
    if ranking == "log_odds":
        rows.sort(key=lambda r: -abs(r.log_odds))
    elif ranking == "jsd_contribution":
        rows.sort(key=lambda r: -r.jsd_contribution)
    else:
        raise ValueError(f"ranking must be 'log_odds' or 'jsd_contribution', got {ranking!r}")
    return rows[:k]


def _group_mean(
    fingerprints: list[Fingerprint],
    group: str,
    group_by: GroupBy,
) -> npt.NDArray[np.float64]:
    """L1-normalized mean of one group's fingerprint distributions."""
    matching = [
        fp.distribution()
        for fp in fingerprints
        if (fp.agent if group_by == "agent" else fp.group) == group
    ]
    if not matching:
        raise ValueError(f"no fingerprints found for group {group!r}")
    stacked = np.stack(matching, axis=0)
    mean = np.asarray(stacked.mean(axis=0), dtype=np.float64)
    total = float(mean.sum())
    if total <= 0.0:
        return mean
    return np.asarray(mean / total, dtype=np.float64)


__all__ = [
    "DiscriminativeProcedure",
    "GroupAtomFrequencies",
    "GroupBy",
    "GroupEntropyStats",
    "Ranking",
    "atom_frequencies_per_group",
    "discriminative_procedures",
    "effective_vocab_size_per_group",
    "entropies_per_group",
]
