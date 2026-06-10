"""Ecosystem catalog: discover trajectory datasets on the Hub and map coverage.

Joins Hub metadata (downloads/likes) with a cheap adapter sniff for every
discovered dataset, so we can see -- across the whole ecosystem, not a curated
list -- which formats dominate and what fraction procgrep can parse today.
This is the data backbone for a browsable index UI.

    python case-studies/ecosystem_catalog.py --top 60 --out catalog.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from procgrep.discover import discover
from procgrep.ingest import plan

# Coarse out-of-scope domains (no discrete tool/code actions to canonicalize).
_OUT_OF_SCOPE = re.compile(
    r"uav|flight|drone|robot|d3il|pusht|sorting|chess|lczero|"
    r"\bmath\b|arcagi|arc-agi|physical|autonomous-vehicle|world-?model",
    re.IGNORECASE,
)


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
        }
        try:
            p = plan(m.id, timeout=25.0)
            row |= {"adapter": p.adapter, "confidence": p.confidence, "supported": p.confidence > 0.0}
        except Exception as e:  # noqa: BLE001 - one bad dataset must not abort the sweep
            row |= {"adapter": None, "confidence": 0.0, "supported": False,
                    "error": f"{type(e).__name__}: {str(e)[:80]}"}
        rows.append(row)
        tag = row.get("adapter") or ("out-of-scope" if row["out_of_scope"] else row.get("error", "?"))
        print(f"{row['downloads']:>8} dl  {row['id']:<58} {tag}", flush=True)

    payload = {"generated": args.generated, "n_discovered": len(metas), "datasets": rows}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    supported = sum(r["supported"] for r in rows)
    by_adapter = Counter(r["adapter"] for r in rows if r["supported"])
    print(f"\ncoverage: {supported}/{len(rows)} datasets parseable")
    for adapter, n in by_adapter.most_common():
        print(f"  {adapter:14s} {n}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
