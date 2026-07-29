"""Ecosystem catalog: discover trajectory datasets on the Hub and map coverage.

Joins Hub metadata (downloads/likes) with a cheap adapter sniff for every
discovered dataset, so we can see -- across the whole ecosystem, not a curated
list -- which formats dominate and what fraction procgrep can parse today.
This is the data backbone for a browsable index UI.

    python interface/ecosystem_catalog.py --top 60 --out catalog.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from procgrep.ingest import plan
from procgrep.ingest.discover import discover

# Coarse out-of-scope domains (no discrete tool/code actions to canonicalize).
_OUT_OF_SCOPE = re.compile(
    r"uav|flight|drone|robot|d3il|pusht|sorting|chess|lczero|"
    r"\bmath\b|arcagi|arc-agi|physical|autonomous-vehicle|world-?model",
    re.IGNORECASE,
)

# Which eval/benchmark a dataset targets (most-specific first so rebench/zero
# win over the generic swe-bench match).
_BENCHMARKS = [
    ("SWE-rebench", r"rebench"),
    ("SWE-smith", r"swe-?smith"),
    ("SWE-Gym", r"swe-?gym"),
    ("SWE-Zero", r"swe-?zero"),
    ("R2E-Gym", r"r2e"),
    ("Terminal-Bench", r"terminal"),
    ("OSWorld", r"osworld"),
    ("WebVoyager", r"webvoyager"),
    ("SWE-bench", r"swe-?bench"),
]


def _benchmark(dataset_id: str) -> str | None:
    for name, pat in _BENCHMARKS:
        if re.search(pat, dataset_id, re.IGNORECASE):
            return name
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=120, help="sniff the top-N by downloads")
    ap.add_argument("--out", default="catalog.json")
    ap.add_argument("--generated", default="", help="ISO timestamp stamped into the catalog")
    args = ap.parse_args()

    metas = discover()
    print(f"discovered {len(metas)} datasets; sniffing top {args.top} by downloads\n", flush=True)

    rows: list[dict] = []
    for m in metas[: args.top]:
        row = {
            "id": m.id,
            "author": m.id.split("/")[0] if "/" in m.id else "",
            "downloads": m.downloads,
            "likes": m.likes,
            "last_modified": m.last_modified[:10] if m.last_modified else "",
            "out_of_scope": bool(_OUT_OF_SCOPE.search(m.id)),
            "benchmark": _benchmark(m.id),
        }
        try:
            p = plan(m.id, timeout=25.0)
            row |= {
                "adapter": p.adapter,
                "confidence": p.confidence,
                "supported": p.confidence > 0.0,
                "candidate": p.candidate,
            }
        except Exception as e:
            row |= {
                "adapter": None,
                "confidence": 0.0,
                "supported": False,
                "candidate": False,
                "error": f"{type(e).__name__}: {str(e)[:80]}",
            }
        rows.append(row)
        tag = row.get("adapter") or (
            "out-of-scope"
            if row["out_of_scope"]
            else ("trace?-unsupported" if row.get("candidate") else "not-a-trace")
        )
        print(f"{row['downloads']:>8} dl  {row['id']:<54} {tag}", flush=True)

    payload = {"generated": args.generated, "n_discovered": len(metas), "datasets": rows}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    # Honest split of the "unsupported" set.
    sup = [r for r in rows if r["supported"]]
    candidates = [r for r in rows if r.get("candidate")]
    fmt_gap = [r for r in candidates if not r["supported"]]
    not_trace = [
        r for r in rows if not r.get("candidate") and not r["supported"] and not r["out_of_scope"]
    ]
    oos = [r for r in rows if r["out_of_scope"]]
    print(f"\nsniffed {len(rows)}:")
    print(f"  parseable      {len(sup)}  {dict(Counter(r['adapter'] for r in sup))}")
    print(f"  format-gap     {len(fmt_gap)}  (trace datasets, adapter missing — the roadmap)")
    print(f"  not-a-trace    {len(not_trace)}  (no conversation column — benchmarks/corpora)")
    print(f"  out-of-scope   {len(oos)}  (non-tool domains)")
    print(
        f"  COVERAGE among trace candidates: {len(sup)}/{len(candidates)} = "
        f"{len(sup) / max(len(candidates), 1):.0%}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
