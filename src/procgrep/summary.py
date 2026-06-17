"""Diff two trace populations by their summary metadata.

procgrep's primary comparison is procedural -- the JSD between procedure
distributions. This is a complementary axis: diff groups by the metadata an
adapter attaches to each trace (verbosity, turn counts, autonomy, tool mix), so
you can contrast *working styles*, not just procedure shapes. It is generic
over whatever numeric metadata a source provides, plus any dict-valued
categorical field (e.g. per-tool call counts), which it contrasts by JSD.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from procgrep.jsd import jsd
from procgrep.types import Trace


@dataclass(frozen=True)
class SummaryDiff:
    """Group-vs-group difference in summary metadata.

    ``deltas`` is ``mean_b - mean_a`` for each numeric metadata key present in
    both groups; ``categorical_jsd`` is the JSD between the two groups'
    distributions for each dict-valued field (0 identical, 1 disjoint).
    """

    label_a: str
    label_b: str
    n_a: int
    n_b: int
    means_a: dict[str, float]
    means_b: dict[str, float]
    deltas: dict[str, float]
    categorical_jsd: dict[str, float]


def _numeric_means(traces: Sequence[Trace]) -> tuple[dict[str, float], set[str]]:
    """Mean of each numeric metadata key over the traces that carry it."""
    sums: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.defaultdict(int)
    for trace in traces:
        for key, value in (trace.metadata or {}).items():
            if isinstance(value, bool):  # bool is an int subclass; not a metric
                continue
            if isinstance(value, (int, float)):
                sums[key] += float(value)
                counts[key] += 1
    return {key: sums[key] / counts[key] for key in sums}, set(counts)


def _categorical_dist(traces: Sequence[Trace], key: str, support: list[str]) -> np.ndarray:
    """Summed counts for a dict-valued metadata field, aligned to ``support``."""
    counter: dict[str, float] = collections.defaultdict(float)
    for trace in traces:
        value = (trace.metadata or {}).get(key)
        if isinstance(value, dict):
            for name, count in value.items():
                if isinstance(count, (int, float)) and not isinstance(count, bool):
                    counter[str(name)] += float(count)
    return np.array([counter.get(name, 0.0) for name in support], dtype=float)


def summary_diff(
    group_a: Sequence[Trace],
    group_b: Sequence[Trace],
    *,
    label_a: str = "a",
    label_b: str = "b",
    categorical: Sequence[str] = ("tools",),
) -> SummaryDiff:
    """Contrast two trace populations by their summary metadata.

    Numeric metadata keys present in both groups are reduced to group means and
    a ``mean_b - mean_a`` delta; each dict-valued ``categorical`` field is
    contrasted by the JSD between the groups' summed distributions.
    """
    means_a, keys_a = _numeric_means(group_a)
    means_b, keys_b = _numeric_means(group_b)
    deltas = {key: round(means_b[key] - means_a[key], 3) for key in sorted(keys_a & keys_b)}

    categorical_jsd: dict[str, float] = {}
    for key in categorical:
        support_set: set[str] = set()
        for group in (group_a, group_b):
            for trace in group:
                value = (trace.metadata or {}).get(key)
                if isinstance(value, dict):
                    support_set.update(str(name) for name in value)
        support = sorted(support_set)
        if not support:
            continue
        dist_a = _categorical_dist(group_a, key, support)
        dist_b = _categorical_dist(group_b, key, support)
        if dist_a.sum() > 0 and dist_b.sum() > 0:
            categorical_jsd[key] = round(float(jsd(dist_a, dist_b)), 6)

    return SummaryDiff(
        label_a=label_a,
        label_b=label_b,
        n_a=len(group_a),
        n_b=len(group_b),
        means_a={key: round(value, 3) for key, value in means_a.items()},
        means_b={key: round(value, 3) for key, value in means_b.items()},
        deltas=deltas,
        categorical_jsd=categorical_jsd,
    )


__all__ = ["SummaryDiff", "summary_diff"]
