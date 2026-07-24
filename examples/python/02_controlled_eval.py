"""Controlled-evaluation workflow.

Hold base model and scaffold fixed; vary one factor (here represented
by the `group` field on each Trace); measure procedural change.

Computes three quantities:

* Across-arm Jensen-Shannon divergence (the signal of interest).
* Within-arm Jensen-Shannon divergence (the noise floor against
  which across-arm JSD is interpreted).
* Leave-one-arm-out predictive-transfer probe.

In a real controlled eval, replace `synthetic_traces.jsonl` with
traces captured per arm (e.g., N=30 traces per temperature setting),
label each by arm via the `group` field, then run this script.

Run from the repository root:

    python examples/python/02_controlled_eval.py
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

from procgrep import (
    canonicalize,
    encode,
    fit_bpe,
    jsd,
    jsd_matrix,
    leave_one_group_out,
)
from procgrep.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "examples" / "data" / "synthetic_traces.jsonl"


def main() -> None:
    traces = canonicalize(list(read_jsonl(TRACES)), adapter="swe-agent")

    # Fit one shared vocabulary across all arms so that the vocabulary
    # itself is not an arm-specific confound.
    vocab = fit_bpe((t.atoms for t in traces), vocab_size=20, seed=0)
    fingerprints = encode(traces, vocab=vocab)

    arms = sorted({fp.group for fp in fingerprints})
    per_arm = Counter(fp.group for fp in fingerprints)
    print(f"arms: {arms}")
    print(f"trajectories per arm: {dict(per_arm)}\n")

    print("across-arm JSD (signal):")
    matrix = jsd_matrix(fingerprints, group_by="group")
    for record in matrix.to_records():
        if str(record["row"]) < str(record["col"]):
            print(f"  {record['row']} vs {record['col']}  JSD = {record['jsd']:.4f}")

    print("\nwithin-arm JSD (noise floor):")
    for arm in arms:
        arm_fps = [fp for fp in fingerprints if fp.group == arm]
        if len(arm_fps) < 2:
            print(f"  {arm}: only {len(arm_fps)} trace, skipping")
            continue
        pairs = [jsd(a.distribution(), b.distribution()) for a, b in combinations(arm_fps, 2)]
        mean = sum(pairs) / len(pairs)
        print(
            f"  {arm}: mean pairwise JSD = {mean:.4f} "
            f"(min={min(pairs):.4f}, max={max(pairs):.4f}, n_pairs={len(pairs)})"
        )

    print("\nleave-one-arm-out probe (predict arm label from fingerprint):")
    if len(arms) < 3:
        print(
            f"  skipped: probe needs at least 3 arms (have {len(arms)}). "
            "When training on K-1 arms, the held-out arm's label is "
            "absent from training; the probe requires that the remaining "
            "K-1 arms supply at least two distinct labels, which fails "
            "with K=2. A real controlled eval typically has 3-8 arms."
        )
    else:
        result = leave_one_group_out(fingerprints, label_field="group", seed=0)
        for arm, acc in sorted(result.per_group_accuracy.items()):
            print(f"  held out {arm}: accuracy {acc:.2f}")


if __name__ == "__main__":
    main()
