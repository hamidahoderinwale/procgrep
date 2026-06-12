"""Intent: precompute the procgrep "spine" store, one whitespace-joined atom
string per canonical trace, across a set of agent-trajectory datasets, and
publish it as a single parquet file (locally and, with --push, to the Hugging
Face dataset repo midah/procgrep-spines) so the Space loader has a fast,
network-cheap artifact to read instead of re-streaming and re-canonicalizing
every dataset on demand.

Design decisions (each with its benefit and its price):

1. Single flat parquet with a fixed 5-column schema
   (dataset, trace_id, agent, task, spine).
   Benefit: a stable contract the Space loader can read with one
   pandas/pyarrow call; columnar + typed so spine scans are fast and the file
   stays small.
   Price: any schema change is a breaking change for every consumer, so the
   column set is frozen here and must be migrated in lockstep.

2. The "spine" is " ".join(trace.atoms) + " " (trailing space included).
   Benefit: a single trailing-space-delimited token stream that substring and
   n-gram tooling can scan uniformly, with every atom (including the last)
   bounded by a space so " edit " matches at the end too.
   Price: it is a denormalized string, not a list, so consumers that want
   structured atoms must split on whitespace and strip the trailing token.

3. One dataset failing must not abort the run; errors are caught, logged, and
   skipped.
   Benefit: a single unparseable or unreachable dataset cannot waste the whole
   (long, outward) precompute; partial output is still useful.
   Price: silent-ish degradation. The store may be missing a dataset, so the
   per-dataset row counts are logged to make a gap visible.

4. Default dataset list is hardcoded to the known-parseable ones, with an
   optional --datasets override and an optional procgrep.discover enumeration.
   Benefit: the common case runs with no arguments and only touches datasets
   we know ingest can handle; discovery stays available for catalog expansion.
   Price: the default list drifts from the Hub over time and must be revisited;
   it is not auto-synced.

5. Push reads the token from the environment or the cached login and never
   prints it.
   Benefit: the same script works locally (cached huggingface-cli login) and in
   CI (HF_TOKEN secret) with no token handling in code paths that log.
   Price: a missing or unauthorized token only surfaces as an upload error at
   the end, not as an upfront check.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Known-parseable agent-trajectory datasets. Deduped: the prompt's list repeats
# SWE-bench/SWE-smith-trajectories, which dict-from-keys collapses while keeping
# first-seen order.
DEFAULT_DATASETS: list[str] = list(
    dict.fromkeys(
        [
            "nebius/SWE-agent-trajectories",
            "ElenaFu/SWE-agent-trajectories",
            "nebius/SWE-rebench-openhands-trajectories",
            "SWE-bench/SWE-smith-trajectories",
            "nvidia/SWE-Zero-openhands-trajectories",
            "SWE-bench/SWE-smith-trajectories",
        ]
    )
)

HF_REPO_ID: str = "midah/procgrep-spines"
SPINE_COLUMNS: list[str] = ["dataset", "trace_id", "agent", "task", "spine"]


def _discover_datasets() -> list[str]:
    """Enumerate candidate datasets via procgrep's Hub discovery.

    Tries the flat ``procgrep.discover`` path first, then the nested
    ``procgrep.ingest.discover`` path, so it works regardless of which the
    installed procgrep exposes.
    """
    try:
        from procgrep.discover import discover  # type: ignore[import-not-found]
    except ImportError:
        from procgrep.ingest.discover import discover

    metas = discover()
    return [m.id for m in metas]


def build_rows(datasets: list[str], cap: int, timeout: float) -> list[dict[str, str]]:
    """Ingest each dataset and emit one spine row per canonical trace.

    A dataset that errors is logged to stderr and skipped so it cannot abort
    the whole run.
    """
    from procgrep.ingest import ingest

    rows: list[dict[str, str]] = []
    for dataset in datasets:
        try:
            traces, _plan = ingest(dataset, limit=cap, timeout=timeout)
        except Exception as exc:  # - one bad dataset must not abort
            print(f"[build_spines] SKIP {dataset}: {exc!r}", file=sys.stderr)
            continue
        for t in traces:
            rows.append(
                {
                    "dataset": dataset,
                    "trace_id": str(t.trace_id),
                    "agent": str(t.agent),
                    "task": str(t.group or ""),
                    "spine": " ".join(t.atoms) + " ",
                }
            )
        print(f"[build_spines] {dataset}: {len(traces)} traces", file=sys.stderr)
    return rows


def write_parquet(rows: list[dict[str, str]], out: str) -> pd.DataFrame:
    """Write the spine rows to ``out`` as parquet with the fixed schema."""
    import pandas as pd

    df = pd.DataFrame(rows, columns=SPINE_COLUMNS)
    df = df.astype(dict.fromkeys(SPINE_COLUMNS, "string"))
    df.to_parquet(out, index=False)
    print(f"[build_spines] wrote {len(df)} rows to {out}", file=sys.stderr)
    return df


def push_to_hub(out: str, repo_id: str = HF_REPO_ID) -> None:
    """Upload the parquet to the HF dataset repo at its root.

    Token is read from HF_TOKEN or the cached login by huggingface_hub; it is
    never read or printed here.
    """
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=out,
        path_in_repo="procgrep_spines.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[build_spines] pushed {out} to {repo_id}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the build_spines command line."""
    parser = argparse.ArgumentParser(
        description="Precompute the procgrep spine parquet store and optionally push it to the Hub.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Explicit dataset ids to ingest. Omit to use the default parseable list; "
        "pass 'discover' as the sole value to enumerate candidates via procgrep.discover.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=20000,
        help="Per-dataset row cap passed to ingest(limit=...).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/procgrep_spines.parquet",
        help="Output parquet path.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help=f"Upload the parquet to the HF dataset repo {HF_REPO_ID}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-dataset ingest timeout in seconds.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the spine store and optionally push it to the Hub."""
    args = parse_args(argv)

    if args.datasets is None:
        datasets: list[str] = list(DEFAULT_DATASETS)
    elif args.datasets == ["discover"]:
        datasets = _discover_datasets()
    else:
        datasets = list(dict.fromkeys(args.datasets))

    print(f"[build_spines] {len(datasets)} datasets, cap={args.cap}", file=sys.stderr)

    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = build_rows(datasets, cap=args.cap, timeout=args.timeout)
    write_parquet(rows, args.out)

    if args.push:
        push_to_hub(args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
