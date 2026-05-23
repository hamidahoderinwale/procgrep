"""Check whether candidate procedural metrics are independent.

A common failure mode in any "we propose K dimensions" paper is
that several of the proposed dimensions move together on real
corpora — when that happens you don't really have K dimensions,
you have fewer-than-K dimensions and some surrogates.

This script computes six procgrep-side group-level metrics and
returns their pairwise Pearson correlation across groups. Pairs
with |r| above ``--threshold`` are flagged as redundant; the
script suggests an independent set by greedily dropping the
weaker member of each over-threshold pair.

Caveat: meaningful correlation needs at least ~10 groups. On the
bundled synthetic corpus there are only 2 groups; the result will
be degenerate (every correlation is +/- 1). Use ``--traces`` and
``--group-by`` to point at a real corpus with many groups (e.g.
84 agents).

Computed metrics:

* mean_traj_length     -- mean number of atoms per trajectory in
                          the group
* effective_vocab      -- effective vocabulary size = exp(entropy
                          of the group-mean procedure distribution)
* mean_entropy         -- mean of per-trajectory procedure entropy
                          across the group
* within_group_jsd     -- mean pairwise JSD between fingerprints
                          inside the group (the "noise floor" of
                          the group)
* procedure_concentration  -- Herfindahl-Hirschman index on the group-
                          mean procedure distribution (sum of squared
                          procedure probabilities; high = mass on few
                          procedures)
* atom_gini            -- Gini coefficient on the per-group atom
                          frequency vector (high = a few atoms
                          dominate the group's behavior)

Run from the repository root:

    python examples/python/08_metric_orthogonality.py
    python examples/python/08_metric_orthogonality.py \\
        --traces my_corpus.jsonl --group-by agent --threshold 0.85
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from procgrep import (
    Fingerprint,
    Trace,
    canonicalize,
    effective_vocab_size_per_group,
    encode,
    entropies_per_group,
    fit_bpe,
    jsd,
)
from procgrep.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = ROOT / "examples" / "synthetic_traces.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES,
        help="JSONL trace file (default: bundled synthetic corpus)",
    )
    parser.add_argument(
        "--adapter",
        default="swe-agent",
        help="canonicalize adapter (default: swe-agent)",
    )
    parser.add_argument(
        "--group-by",
        default="group",
        help="Trace.metadata key (or 'group' or 'agent') to partition by (default: group)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50,
        help="BPE target vocabulary size (default: 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="absolute correlation above which two metrics are deemed redundant",
    )
    return parser.parse_args()


def regroup(traces: list[Trace], key: str) -> list[Trace]:
    """Build new traces with `group` derived from `key`.

    `key` can be the string 'agent', 'group' (the existing field),
    or any `metadata` field. Traces missing the key are skipped.
    """
    out: list[Trace] = []
    for t in traces:
        if key == "agent":
            label: str | None = t.agent
        elif key == "group":
            label = t.group
        else:
            value = t.metadata.get(key)
            label = None if value is None else str(value)
        if label is None:
            continue
        out.append(replace(t, group=label))
    return out


def gini(values: list[float]) -> float:
    """Gini coefficient on a non-negative vector. 0 = perfectly even, 1 = one bin owns everything."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def compute_metrics(
    traces: list[Trace], fingerprints: list[Fingerprint]
) -> dict[str, dict[str, float]]:
    """Compute six per-group metrics. Returns metric -> {group: value}."""
    groups = sorted({fp.group for fp in fingerprints})
    fps_by_group: dict[str, list[Fingerprint]] = {g: [] for g in groups}
    traces_by_group: dict[str, list[Trace]] = {g: [] for g in groups}
    for fp in fingerprints:
        fps_by_group[fp.group].append(fp)
    for t in traces:
        if t.group in traces_by_group:
            traces_by_group[t.group].append(t)

    mean_traj_length = {g: np.mean([len(t.atoms) for t in traces_by_group[g]]) for g in groups}
    eff_vocab = effective_vocab_size_per_group(fingerprints, group_by="group")
    entropies = entropies_per_group(fingerprints, group_by="group")
    mean_entropy = {g: entropies[g].median for g in groups}

    within_jsd: dict[str, float] = {}
    for g in groups:
        fps = fps_by_group[g]
        if len(fps) < 2:
            within_jsd[g] = 0.0
            continue
        pairs = [jsd(a.distribution(), b.distribution()) for a, b in combinations(fps, 2)]
        within_jsd[g] = float(np.mean(pairs))

    procedure_concentration: dict[str, float] = {}
    for g in groups:
        fps = fps_by_group[g]
        mean_dist = np.mean([fp.distribution() for fp in fps], axis=0)
        procedure_concentration[g] = float(np.sum(mean_dist**2))

    atom_gini: dict[str, float] = {}
    for g in groups:
        atom_counts: Counter[str] = Counter()
        for t in traces_by_group[g]:
            atom_counts.update(t.atoms)
        atom_gini[g] = gini(list(atom_counts.values()))

    return {
        "mean_traj_length": {g: float(v) for g, v in mean_traj_length.items()},
        "effective_vocab": {g: float(v) for g, v in eff_vocab.items()},
        "mean_entropy": {g: float(v) for g, v in mean_entropy.items()},
        "within_group_jsd": within_jsd,
        "procedure_concentration": procedure_concentration,
        "atom_gini": atom_gini,
    }


def correlation_matrix(metrics: dict[str, dict[str, float]]) -> tuple[list[str], np.ndarray]:
    """Pairwise Pearson correlation between metric columns, evaluated over groups."""
    names = list(metrics.keys())
    groups = sorted(next(iter(metrics.values())).keys())
    matrix = np.array([[metrics[m][g] for g in groups] for m in names], dtype=np.float64)
    if matrix.shape[1] < 2:
        return names, np.full((len(names), len(names)), np.nan)
    return names, np.corrcoef(matrix)


def independent_set(
    names: list[str], corr: np.ndarray, threshold: float
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedily drop the metric in each redundant pair that has higher mean |r| with the rest.

    Returns (kept_metrics, redundant_pairs).
    """
    n = len(names)
    redundant_pairs: list[tuple[str, str, float]] = []
    dropped: set[str] = set()
    for i in range(n):
        for j in range(i + 1, n):
            r = corr[i, j]
            if np.isnan(r):
                continue
            if abs(r) >= threshold:
                redundant_pairs.append((names[i], names[j], float(r)))
                mean_i = float(np.nanmean(np.abs(np.delete(corr[i], i))))
                mean_j = float(np.nanmean(np.abs(np.delete(corr[j], j))))
                victim = names[i] if mean_i >= mean_j else names[j]
                dropped.add(victim)
    kept = [m for m in names if m not in dropped]
    return kept, redundant_pairs


def main() -> None:
    args = parse_args()
    raw = list(read_jsonl(args.traces))
    traces = canonicalize(raw, adapter=args.adapter)
    traces = regroup(traces, args.group_by)

    if not traces:
        print(f"no traces had a non-null '{args.group_by}'; nothing to do")
        return

    groups = sorted({t.group for t in traces})
    print(f"loaded {len(traces)} traces grouped by '{args.group_by}': {len(groups)} groups")
    if len(groups) < 10:
        print(f"  warning: {len(groups)} groups is too few for meaningful correlation;")
        print("  results below should be treated as a smoke test, not a finding.")

    vocab = fit_bpe((t.atoms for t in traces), vocab_size=args.vocab_size, seed=0)
    fps = encode(traces, vocab=vocab)

    metrics = compute_metrics(traces, fps)

    print("\nper-group metric values:")
    metric_names = list(metrics.keys())
    header = "  " + f"{'group':<24s}" + "".join(f"{m:>22s}" for m in metric_names)
    print(header)
    for g in groups:
        row = f"  {g:<24s}" + "".join(f"{metrics[m][g]:>22.4f}" for m in metric_names)
        print(row)

    names, corr = correlation_matrix(metrics)
    print("\npairwise Pearson correlation (off-diagonal):")
    print("  " + f"{'':<22s}" + "".join(f"{n[:18]:>20s}" for n in names))
    for i, name_i in enumerate(names):
        row_cells = []
        for j in range(len(names)):
            r = corr[i, j]
            cell = "      ." if (i == j or np.isnan(r)) else f"{r:>20.3f}"
            row_cells.append(cell)
        print(f"  {name_i:<22s}" + "".join(row_cells))

    kept, redundant_pairs = independent_set(names, corr, args.threshold)
    print(f"\nat threshold |r| >= {args.threshold:.2f}:")
    if not redundant_pairs:
        print("  no pair exceeds the redundancy threshold")
    else:
        print("  redundant pairs:")
        for a, b, r in redundant_pairs:
            print(f"    {a} <-> {b}: r = {r:+.3f}")
    print(f"  suggested independent set ({len(kept)} of {len(names)}): {kept}")


if __name__ == "__main__":
    main()
