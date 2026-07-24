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
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from procgrep.bpe import ProcedureVocabulary
from procgrep.encode import encode
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


def _categorical_dist(
    traces: Sequence[Trace], key: str, support: list[str]
) -> npt.NDArray[np.float64]:
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


def variance_decomposition(
    traces: Sequence[Trace],
    vocab: ProcedureVocabulary,
    factors: Sequence[str],
    *,
    sample: int = 800,
    seed: int = 0,
) -> dict[str, float]:
    """Pseudo-R2 of how much procedural variation each factor explains.

    A PERMANOVA-style partition over the pairwise JSD-squared between trajectory
    fingerprints: for each factor (a trace-metadata key), R2 = 1 - within-group
    dispersion / total dispersion. R2 near 1 means traces sharing that factor's
    value are procedurally homogeneous, so the factor explains a lot; near 0
    means it does not. Distances are JSD between encoded fingerprints, so this
    answers "what drives how an agent works -- the task, the model, the
    scaffold?" on one model-free metric. O(N^2); subsamples to ``sample``.
    """
    items = list(traces)
    if len(items) > sample:
        items = random.Random(seed).sample(items, sample)
    n = len(items)
    if n < 3:
        raise ValueError("variance_decomposition needs at least 3 traces")
    dists = [np.asarray(fp.distribution(), dtype=np.float64) for fp in encode(items, vocab=vocab)]
    sq: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = jsd(dists[i], dists[j])
            sq[(i, j)] = d * d
    ss_total = sum(sq.values()) / n
    if ss_total <= 0:
        return dict.fromkeys(factors, 0.0)

    out: dict[str, float] = {}
    for factor in factors:
        groups: dict[object, list[int]] = collections.defaultdict(list)
        for idx, trace in enumerate(items):
            groups[(trace.metadata or {}).get(factor)].append(idx)
        ss_within = 0.0
        for members in groups.values():
            if len(members) < 2:
                continue
            pair_sum = sum(
                sq[(members[a], members[b])]
                for a in range(len(members))
                for b in range(a + 1, len(members))
            )
            ss_within += pair_sum / len(members)
        out[factor] = round(1.0 - ss_within / ss_total, 4)
    return out


def autonomy_runlength(
    traces: Sequence[Trace], *, prompt_atom: str = "prompt_ai"
) -> dict[str, float]:
    """Distribution of agent-action run lengths between human prompts.

    A run is the count of non-prompt atoms following a human prompt; a trace with
    no human prompt at all counts as one fully-autonomous run (its whole action
    length -- one instruction, then the agent goes). A high run-length means
    agent-dense, let-it-iterate work; a low one means tight, interleaved human
    steering. Model-free and deterministic over the atom stream -- the "autonomy"
    axis (let-it-iterate vs interrupt) as a measurable quantity, not a vibe.

    Returns ``n_runs`` and the run-length ``mean`` / ``median`` / ``p90`` / ``max``.
    """
    runs: list[int] = []
    for trace in traces:
        atoms = list(trace.atoms)
        if prompt_atom not in atoms:
            if atoms:
                runs.append(len(atoms))
            continue
        current = 0
        seen = False
        for atom in atoms:
            if atom == prompt_atom:
                if seen:
                    runs.append(current)
                seen = True
                current = 0
            else:
                current += 1
        if seen:
            runs.append(current)
    if not runs:
        return {"n_runs": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(runs)
    mid = len(ordered) // 2
    median = float(ordered[mid]) if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n_runs": float(len(runs)),
        "mean": round(sum(runs) / len(runs), 2),
        "median": median,
        "p90": float(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]),
        "max": float(max(runs)),
    }


__all__ = ["SummaryDiff", "autonomy_runlength", "summary_diff", "variance_decomposition"]
