"""Group-level descriptive and discriminative statistics.

A convenience layer above the core fingerprinting pipeline
(`canonicalize`, `fit_bpe`, `encode`, `jsd`) that answers four
questions every descriptive table and every cross-group comparison
hits:

* `atom_frequencies_per_group`: which raw atoms dominate each group?
* `effective_vocab_size_per_group`: how diverse is each group's
  procedural vocabulary, expressed as the equivalent number of
  uniformly-used motifs?
* `entropies_per_group`: how diverse is the typical trajectory inside
  each group?
* `discriminative_motifs`: which motifs most strongly distinguish two
  groups, ranked by log-odds ratio or by Jensen-Shannon-divergence
  contribution?

The helpers are pure functions of the existing `Trace`, `Fingerprint`,
and `MotifVocabulary` types. No new dependencies, no breaking
changes to other modules. Suitable for descriptive tables in papers
and for controlled-eval summaries where the within-arm-vs-across-arm
question dominates.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from procgrep.bpe import MotifVocabulary
from procgrep.encode import Fingerprint
from procgrep.types import Atom, Trace

GroupBy = Literal["group", "agent"]
Ranking = Literal["log_odds", "jsd_contribution"]


@dataclass(frozen=True)
class GroupAtomFrequencies:
    """Top-K most-frequent raw atoms in one group.

    Attributes:
        group: The group label this entry summarizes.
        n_trajectories: Number of trajectories aggregated.
        n_atoms_total: Sum of atom counts across all trajectories.
        top: Ordered tuple of ``(atom, count, per_trajectory_rate)``
            entries, length up to ``k``.
    """

    group: str
    n_trajectories: int
    n_atoms_total: int
    top: tuple[tuple[Atom, int, float], ...]


@dataclass(frozen=True)
class GroupEntropyStats:
    """Per-trajectory Shannon entropy summary for one group.

    Attributes:
        group: The group label this entry summarizes.
        n: Number of trajectories in the group.
        median: Median per-trajectory entropy, in nats.
        q1: Twenty-fifth percentile.
        q3: Seventy-fifth percentile.
        min: Minimum per-trajectory entropy.
        max: Maximum per-trajectory entropy.
    """

    group: str
    n: int
    median: float
    q1: float
    q3: float
    min: float
    max: float


@dataclass(frozen=True)
class DiscriminativeMotif:
    """One motif that distinguishes two groups.

    Attributes:
        motif: The token (atom or BPE-merged motif).
        p_a: Group-mean probability of the motif in group A.
        p_b: Group-mean probability of the motif in group B.
        log_odds: ``log((p_a + eps) / (p_b + eps))``. Positive
            means more common in group A; negative means more
            common in group B.
        jsd_contribution: This motif's contribution to the JSD
            between the two group-mean distributions. Always
            non-negative.
    """

    motif: str
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
    """Top-K most-frequent raw atoms in each group.

    Operates on raw atom sequences (`Trace.atoms`), not on BPE
    motifs. For motif-level frequencies use the fingerprint counts
    directly.

    Args:
        traces: Trajectories to aggregate. Materialized internally.
        k: How many top atoms to keep per group.
        group_by: ``"group"`` to use `Trace.grouping()`, ``"agent"``
            to override and use `Trace.agent`.

    Returns:
        Mapping from group label to `GroupAtomFrequencies`.
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

    Defined as ``exp(Shannon entropy of the group-mean motif
    distribution)``. Interpretation: a group with effective vocab
    ``N`` uses a procedural vocabulary equivalent in diversity to
    a uniform distribution over ``N`` motifs. Equivalently, the
    perplexity of the group-mean distribution under itself.

    Args:
        fingerprints: Fingerprints to aggregate. Materialized
            internally.
        group_by: ``"group"`` or ``"agent"``.

    Returns:
        Mapping from group label to effective vocab size (float >= 1.0).
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

    Computes each fingerprint's entropy via `Fingerprint.entropy()`,
    then summarizes by group with median, IQR, and range. Useful
    for surfacing groups whose typical trajectory is monolithic
    (low entropy) versus diverse (high entropy).

    Args:
        fingerprints: Fingerprints to summarize.
        group_by: ``"group"`` or ``"agent"``.

    Returns:
        Mapping from group label to `GroupEntropyStats`.
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


def discriminative_motifs(
    fingerprints: Iterable[Fingerprint],
    vocab: MotifVocabulary,
    *,
    group_a: str,
    group_b: str,
    k: int = 10,
    ranking: Ranking = "log_odds",
    epsilon: float = 1e-6,
    group_by: GroupBy = "group",
) -> list[DiscriminativeMotif]:
    """Top-K motifs distinguishing ``group_a`` from ``group_b``.

    Computes group-mean motif distributions for both groups, then
    ranks every motif by either its log-odds ratio or its
    contribution to the Jensen-Shannon divergence between the two
    group means. Returns the top ``k`` by the chosen ranking.

    Args:
        fingerprints: Fingerprints from at least the two named
            groups. Other groups are ignored.
        vocab: The vocabulary the fingerprints were encoded under.
            The function checks that vocab size matches fingerprint
            dimension.
        group_a, group_b: The two group labels to compare.
        k: Number of motifs to return.
        ranking: ``"log_odds"`` sorts by absolute log-odds (motifs
            with the largest A-vs-B ratio in either direction).
            ``"jsd_contribution"`` sorts by the motif's contribution
            to JSD (always non-negative).
        epsilon: Smoothing constant for log-odds to avoid log(0).
        group_by: ``"group"`` or ``"agent"``.

    Returns:
        List of `DiscriminativeMotif` of length up to ``k``.
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
    rows: list[DiscriminativeMotif] = []
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
            DiscriminativeMotif(
                motif=token,
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
    """L1-normalized mean of distributions for one group's fingerprints."""
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
    "DiscriminativeMotif",
    "GroupAtomFrequencies",
    "GroupBy",
    "GroupEntropyStats",
    "Ranking",
    "atom_frequencies_per_group",
    "discriminative_motifs",
    "effective_vocab_size_per_group",
    "entropies_per_group",
]
