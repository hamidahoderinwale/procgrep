"""Empirical redundancy study: curate a public HF trajectory dataset.

Dynamically ingests <dataset> via ``procgrep.ingest`` (schema-sniffed, no
hard-coded dataset->adapter table), then reports structural redundancy and the
diversity-vs-shortest coverage gap from ``procgrep.curate``. CPU-only; no model.

    python case-studies/curate_hf.py SWE-bench/SWE-smith-trajectories --limit 3000 --target 800
"""

from __future__ import annotations

import argparse

from procgrep.curate import curate
from procgrep.ingest import ingest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--limit", type=int, default=3000, help="rows to stream")
    ap.add_argument("--target", type=int, default=None, help="diverse-subset size")
    ap.add_argument("--vocab-size", type=int, default=128)
    ap.add_argument("--near-dup-jsd", type=float, default=0.05)
    args = ap.parse_args()

    print(f"ingesting up to {args.limit} rows from {args.dataset} …", flush=True)
    traces, plan = ingest(args.dataset, limit=args.limit)
    print(plan.summary())
    print(f"\ningested {len(traces)} traces "
          f"({sum(1 for t in traces if t.atoms)} non-empty)\n")

    report = curate(
        traces,
        vocab_size=args.vocab_size,
        near_dup_jsd=args.near_dup_jsd,
        target_size=args.target,
    )
    print(report.summary())


if __name__ == "__main__":
    main()
