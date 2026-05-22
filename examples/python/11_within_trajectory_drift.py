"""Within-trajectory procedural drift: does an agent's procedure shift over time?

The question this script answers:

    Every other procgrep analysis treats a trajectory as a single
    motif distribution: one fingerprint per trace. That assumes the
    procedure is *stationary* over the course of the trajectory.
    But agents may have a "warm-up procedure" (early atoms) that
    differs from their "steady-state procedure" (later atoms), or
    they may visibly degrade as the context grows. Does the
    procedural fingerprint change as we walk through the trace?

The setup: slice each trace into three equal-length slices
(``prefix``, ``middle``, ``suffix``), encode each slice as a
fingerprint with the same shared vocabulary, then ask three things.

1. **JSD between slice positions.** Mean fingerprint per slice
   position; pairwise JSD. If the procedure drifts, prefix vs
   suffix JSD is meaningfully above zero.
2. **LOGO probe with slice-position as the label.** A classifier
   trained on prefix+middle predicts the held-out suffix's slice
   position above chance iff suffix has a distinguishable signature.
3. **Top discriminative atoms** between prefix and suffix per agent.
   Surfaces the actual atoms that distinguish early-trajectory from
   late-trajectory procedure.

Two falsifiers:

* JSD between slice positions at the noise floor (close to the JSD
  between random splits of the same slice). No drift.
* LOGO probe at chance (1/3 for three slice positions). The slice
  position is not learnable.

The script runs on the bundled gumtree fixture by default; the
analysis is language-/adapter-agnostic.

Run from the repository root:

    python examples/python/11_within_trajectory_drift.py
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from procgrep import (
    Trace,
    canonicalize,
    encode,
    fit_bpe,
    jsd,
    leave_one_group_out,
)
from procgrep.io import read_jsonl
from procgrep.types import AtomSequence

warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = ROOT / "examples" / "synthetic_gumtree_traces.jsonl"

SLICE_POSITIONS = ("prefix", "middle", "suffix")
MIN_ATOMS_PER_SLICE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES,
        help="JSONL trace file (default: bundled gumtree fixture)",
    )
    parser.add_argument(
        "--adapter",
        default="gumtree",
        help="canonicalize adapter (default: gumtree)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50,
        help="BPE target vocabulary size (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the LOGO classifier (default: 0)",
    )
    return parser.parse_args()


def split_into_thirds(atoms: AtomSequence) -> tuple[AtomSequence, AtomSequence, AtomSequence]:
    """Partition an atom sequence into prefix / middle / suffix thirds.

    For sequences shorter than 3 * MIN_ATOMS_PER_SLICE, returns the
    full sequence in the prefix slot and empty slots elsewhere; the
    caller is expected to filter such traces out of drift analysis.
    """
    n = len(atoms)
    if n < 3 * MIN_ATOMS_PER_SLICE:
        return list(atoms), [], []
    boundary_1 = n // 3
    boundary_2 = 2 * n // 3
    return (
        list(atoms[:boundary_1]),
        list(atoms[boundary_1:boundary_2]),
        list(atoms[boundary_2:]),
    )


def slice_traces(traces: list[Trace]) -> list[Trace]:
    """Produce one Trace per (original trace, slice position) pair.

    Each slice keeps the original ``agent``, sets ``group`` to the
    slice position, and uses a derived trace id.
    """
    out: list[Trace] = []
    for t in traces:
        prefix, middle, suffix = split_into_thirds(t.atoms)
        for position, atoms in zip(SLICE_POSITIONS, (prefix, middle, suffix), strict=True):
            if len(atoms) < MIN_ATOMS_PER_SLICE:
                continue
            out.append(
                Trace(
                    trace_id=f"{t.trace_id}#{position}",
                    agent=t.agent,
                    group=position,
                    atoms=atoms,
                    metadata={"original_trace_id": t.trace_id, "slice_position": position},
                )
            )
    return out


def top_atoms(atoms_iter: list[AtomSequence], k: int = 5) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for seq in atoms_iter:
        counter.update(seq)
    return counter.most_common(k)


def main() -> None:
    args = parse_args()
    raw = list(read_jsonl(args.traces))
    traces = canonicalize(raw, adapter=args.adapter)
    print(f"loaded {len(traces)} full trajectories")

    sliced = slice_traces(traces)
    print(
        f"produced {len(sliced)} slice-traces "
        f"({sum(1 for s in sliced if s.group == 'prefix')} prefix, "
        f"{sum(1 for s in sliced if s.group == 'middle')} middle, "
        f"{sum(1 for s in sliced if s.group == 'suffix')} suffix)"
    )
    if not sliced:
        raise SystemExit("no traces long enough to slice into thirds; aborting.")

    # --- 1. Mean-fingerprint JSD between slice positions --------------------
    vocab = fit_bpe((s.atoms for s in sliced), vocab_size=args.vocab_size, seed=args.seed)
    fps = encode(sliced, vocab=vocab)

    by_position: dict[str, list[np.ndarray]] = {p: [] for p in SLICE_POSITIONS}
    for fp in fps:
        by_position[fp.group].append(fp.distribution())
    means = {p: np.mean(arr, axis=0) for p, arr in by_position.items() if arr}

    print("\nmean-fingerprint JSD between slice positions:")
    print(f"  {'pair':24s} {'JSD':>8s}")
    for i, a in enumerate(SLICE_POSITIONS):
        for b in SLICE_POSITIONS[i + 1 :]:
            if a in means and b in means:
                value = jsd(means[a], means[b])
                print(f"  {a:>10s} vs {b:<10s}  {value:>8.4f}")

    # --- 2. LOGO probe with slice-position as label -------------------------
    # `group` IS the label here, so leave-one-group-out using `label_field="group"`
    # would tautologically train on examples whose label is never the held-out
    # label. Instead we use the original trace id as the group, so each fold
    # holds out all three slices of one full trajectory and the classifier has
    # to learn the slice-position label from the other trajectories.
    probe_traces: list[Trace] = []
    slice_label_by_trace_id: dict[str, str] = {}
    for s in sliced:
        original_id = str(s.metadata["original_trace_id"])
        probe_traces.append(
            Trace(
                trace_id=s.trace_id,
                agent=s.agent,
                # Each fold holds out one full trajectory's three slices.
                group=original_id,
                atoms=s.atoms,
            )
        )
        slice_label_by_trace_id[s.trace_id] = str(s.metadata["slice_position"])
    probe_fps = encode(probe_traces, vocab=vocab)

    # Manual LOGO with slice-position as the prediction target.
    x = np.stack([fp.distribution() for fp in probe_fps], axis=0)
    labels = np.array([slice_label_by_trace_id[fp.trace_id] for fp in probe_fps])
    group_arr = np.array([fp.group for fp in probe_fps])

    accs: list[float] = []
    for train_idx, test_idx in LeaveOneGroupOut().split(x, labels, group_arr):
        if len(set(labels[train_idx])) < 2:
            continue
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=args.seed
        )
        clf.fit(x[train_idx], labels[train_idx])
        preds = clf.predict(x[test_idx])
        accs.append(float(np.mean(preds == labels[test_idx])))

    if accs:
        overall = float(np.mean(accs))
        print(
            f"\nLOGO probe — predict slice-position from fingerprint "
            f"(chance ≈ 1/{len(SLICE_POSITIONS)} = {1 / len(SLICE_POSITIONS):.2f}):"
        )
        print(f"  overall accuracy across {len(accs)} folds: {overall:.2f}")

    # Also report the standard LOGO with agent labels, treating slice
    # position as the group — useful for asking "does attribution
    # transfer across held-out slice positions?"
    cross_slice = leave_one_group_out(fps, label_field="agent", seed=args.seed)
    print(
        "\nattribution-transfer across slice positions "
        "(predict agent label on held-out slice position):"
    )
    print(f"  overall accuracy: {cross_slice.overall_accuracy:.2f}")
    for pos in SLICE_POSITIONS:
        if pos in cross_slice.per_group_accuracy:
            print(f"  held-out {pos:>8s}: {cross_slice.per_group_accuracy[pos]:.2f}")

    # --- 3. Top atoms by slice position -------------------------------------
    print("\ntop atoms by slice position (raw counts):")
    for pos in SLICE_POSITIONS:
        seqs = [s.atoms for s in sliced if s.group == pos]
        if not seqs:
            continue
        ranked = top_atoms(seqs, k=5)
        rendered = ", ".join(f"{a}={c}" for a, c in ranked)
        print(f"  {pos:>8s}: {rendered}")


if __name__ == "__main__":
    main()
