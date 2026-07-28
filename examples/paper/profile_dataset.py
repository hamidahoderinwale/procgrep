"""Per-dataset profile for the L2/L3 interface.

Emits one profile.json for a dataset: overall redundancy (curate), a by-model
breakdown (n, length, dup, action-mix), a length distribution, and a few
sampled traces with their aligned conversations (atom spine + raw turns).

    python case-studies/profile_dataset.py nebius/SWE-agent-trajectories --limit 2000 --out profile.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from procgrep.curate import curate
from procgrep.ingest import ingest
from procgrep.types import Trace

# Canonical atoms, in display order.
COARSE = [
    "search_repo", "read_file", "edit", "create_file", "run_test",
    "submit", "think", "localize", "delete_file", "error", "other",
]


def _pct(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[i]


def _model_stats(traces: list[Trace]) -> dict:
    lens = sorted(len(t.atoms) for t in traces)
    uniq = len({tuple(t.atoms) for t in traces})
    mix: Counter[str] = Counter(a for t in traces for a in t.atoms)
    total = sum(mix.values()) or 1
    return {
        "n": len(traces),
        "median_len": _pct(lens, 0.5),
        "p10_len": _pct(lens, 0.1),
        "p90_len": _pct(lens, 0.9),
        "exact_dup_rate": round(1 - uniq / len(traces), 3) if traces else 0.0,
        "action_mix": {a: round(mix[a] / total, 3) for a in COARSE if mix.get(a)},
    }


def _turn_preview(turn: object) -> dict | None:
    if not isinstance(turn, dict):
        return None
    role = turn.get("role")
    tool_calls = turn.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        names = [
            (tc.get("function") or {}).get("name")
            for tc in tool_calls
            if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
        ]
        return {"role": role, "tools": [n for n in names if n]}
    text = turn.get("content") or turn.get("text") or ""
    if isinstance(text, str):
        text = " ".join(text.split())[:180]
    return {"role": role, "text": text}


def _sample(trace: Trace) -> dict:
    turns = trace.metadata.get("messages") if isinstance(trace.metadata, dict) else None
    previews = []
    if isinstance(turns, list):
        previews = [p for t in turns if (p := _turn_preview(t)) is not None][:60]
    return {
        "trace_id": trace.trace_id,
        "model": trace.agent,
        "atoms": list(trace.atoms),
        "turns": previews,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out", default="profile.json")
    args = ap.parse_args()

    traces, plan = ingest(args.dataset, limit=args.limit)
    print(plan.summary(), flush=True)
    report = curate(traces, target_size=min(args.limit // 2, 800))

    by_model: dict[str, list[Trace]] = defaultdict(list)
    for t in traces:
        by_model[t.agent].append(t)
    models = {m: _model_stats(ts) for m, ts in sorted(by_model.items(), key=lambda kv: -len(kv[1]))}

    lens = sorted(len(t.atoms) for t in traces)
    sample_idx = report.subset_indices[: args.samples] or list(range(min(args.samples, len(traces))))

    profile = {
        "dataset": args.dataset,
        "adapter": plan.adapter,
        "n_traces": len(traces),
        "n_models": len(models),
        "redundancy": {
            "exact_dup_rate": report.exact_duplicate_rate,
            "near_dup_rate": report.near_duplicate_rate,
            "coverage_shortest": report.coverage_shortest,
            "coverage_diverse": report.coverage_diverse,
        },
        "length": {
            "median": _pct(lens, 0.5), "p10": _pct(lens, 0.1), "p90": _pct(lens, 0.9),
            "max": lens[-1] if lens else 0,
        },
        "by_model": models,
        "samples": [_sample(traces[i]) for i in sample_idx],
        "atoms_pool": [
            {"trace_id": traces[i].trace_id, "model": traces[i].agent, "atoms": list(traces[i].atoms)}
            for i in list(dict.fromkeys(report.subset_indices + list(range(len(traces)))))[:500]
        ],
    }
    with open(args.out, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n{args.dataset}: {len(traces)} traces, {len(models)} models")
    for m, s in list(models.items())[:8]:
        top = " ".join(f"{a}={v}" for a, v in list(s["action_mix"].items())[:4])
        print(f"  {m[:40]:40s} n={s['n']:>4} med_len={s['median_len']:>3} dup={s['exact_dup_rate']:.0%}  {top}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
