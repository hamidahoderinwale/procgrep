"""Group-level descriptive and discriminative statistics.

Demonstrates the four helpers introduced in v0.1.1 stats module:

* `atom_frequencies_per_group`: which raw atoms dominate each group.
* `effective_vocab_size_per_group`: a group's procedural-vocabulary
  diversity, expressed as the equivalent number of uniformly-used procedures.
* `entropies_per_group`: per-trajectory Shannon entropy summarized
  by group.
* `discriminative_procedures`: top procedures separating two groups, ranked
  by log-odds (default) or by Jensen-Shannon-divergence contribution.

Run from the repository root:

    python examples/python/03_discriminative_procedures.py
"""

from __future__ import annotations

from pathlib import Path

from procgrep import (
    atom_frequencies_per_group,
    canonicalize,
    discriminative_procedures,
    effective_vocab_size_per_group,
    encode,
    entropies_per_group,
    fit_bpe,
)
from procgrep.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "examples" / "synthetic_traces.jsonl"


def main() -> None:
    traces = canonicalize(list(read_jsonl(TRACES)), adapter="swe-agent")
    vocab = fit_bpe((t.atoms for t in traces), vocab_size=20, seed=0)
    fingerprints = encode(traces, vocab=vocab)

    print("=" * 64)
    print("top-K raw atoms per group")
    print("=" * 64)
    freqs = atom_frequencies_per_group(traces, k=5, group_by="group")
    for group in sorted(freqs):
        entry = freqs[group]
        print(f"\n{group}  (n_trajectories={entry.n_trajectories})")
        for atom, count, rate in entry.top:
            print(f"  {atom:15s} count={count:4d}  per_traj={rate:.2f}")

    print("\n" + "=" * 64)
    print("effective vocabulary size per group")
    print("=" * 64)
    eff = effective_vocab_size_per_group(fingerprints, group_by="group")
    for group in sorted(eff):
        print(f"  {group:15s} effective_vocab = {eff[group]:.2f}")

    print("\n" + "=" * 64)
    print("per-trajectory entropy summary per group")
    print("=" * 64)
    ents = entropies_per_group(fingerprints, group_by="group")
    for group in sorted(ents):
        s = ents[group]
        print(f"  {group:15s} n={s.n}  median={s.median:.3f}  IQR=[{s.q1:.3f}, {s.q3:.3f}]")

    print("\n" + "=" * 64)
    print("top discriminative procedures: control vs treatment")
    print("=" * 64)
    top = discriminative_procedures(
        fingerprints,
        vocab,
        group_a="control",
        group_b="treatment",
        k=5,
        ranking="log_odds",
        group_by="group",
    )
    print(f"  {'procedure':35s} {'p_a':>8s} {'p_b':>8s} {'log_odds':>10s}")
    for m in top:
        print(f"  {m.procedure:35s} {m.p_a:8.3f} {m.p_b:8.3f} {m.log_odds:10.3f}")


if __name__ == "__main__":
    main()
