"""Ecosystem redundancy study: run procgrep curate across public HF trajectory
datasets and tabulate structural redundancy + the diversity-vs-shortest gap.

Dynamic ingestion (schema-sniffed) means one command works on every dataset
regardless of scaffold/format; per-dataset failures are recorded, not fatal.
CPU-only; no model.

    python case-studies/redundancy_study.py --limit 3000 --target 800 --out study.json
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

from procgrep.curate import curate
from procgrep.ingest import ingest

DATASETS = [
    "SWE-bench/SWE-smith-trajectories",
    "SWE-Gym/OpenHands-Sampled-Trajectories",
    "nvidia/SWE-Hero-openhands-trajectories",
    "nebius/SWE-rebench-openhands-trajectories",
    "nebius/SWE-agent-trajectories",
    "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k",
]


def run_one(ds: str, limit: int, target: int) -> dict:
    t0 = time.time()
    traces, plan = ingest(ds, limit=limit)
    rep = curate(traces, target_size=target)
    return {
        "dataset": ds,
        "adapter": plan.adapter,
        "confidence": round(plan.confidence, 2),
        "n": rep.n_traces,
        "nonempty": sum(1 for t in traces if t.atoms),
        "exact_dup_rate": round(rep.exact_duplicate_rate, 3),
        "near_dup_rate": round(rep.near_duplicate_rate, 3),
        "coverage_diverse": round(rep.coverage_diverse, 3),
        "coverage_shortest": round(rep.coverage_shortest, 3),
        "coverage_random": round(rep.coverage_random, 3),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--target", type=int, default=800)
    ap.add_argument("--out", default="redundancy_study.json")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()

    rows: list[dict] = []
    for ds in args.datasets:
        print(f"\n### {ds}", flush=True)
        try:
            r = run_one(ds, args.limit, args.target)
            rows.append(r)
            print(json.dumps(r), flush=True)
        except Exception as e:  # noqa: BLE001 - one bad dataset must not abort the sweep
            print(f"FAILED {ds}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            rows.append({"dataset": ds, "error": f"{type(e).__name__}: {e}"})

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n" + "=" * 100)
    hdr = (f"{'dataset':44s} {'adapter':10s} {'n':>5s} {'exact':>6s} "
           f"{'near':>6s} {'divers':>6s} {'short':>6s} {'rand':>6s}")
    print(hdr)
    for r in rows:
        if "error" in r:
            print(f"{r['dataset']:44s} ERROR  {r['error'][:48]}")
            continue
        print(f"{r['dataset']:44s} {r['adapter']:10s} {r['n']:>5d} "
              f"{r['exact_dup_rate']:>6.1%} {r['near_dup_rate']:>6.1%} "
              f"{r['coverage_diverse']:>6.0%} {r['coverage_shortest']:>6.0%} "
              f"{r['coverage_random']:>6.0%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
