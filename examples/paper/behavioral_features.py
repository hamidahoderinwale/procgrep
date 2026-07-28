"""Exploration vs exploitation and behavioral zeitgeist signals.

Extracts per-trajectory features beyond simple atom fractions:
  - search_first: does the agent search before its first edit? (exploration-first)
  - edit_run_max: longest consecutive edit streak (batch-style vs interleaved)
  - think_early_frac: fraction of think atoms in first 25% of trajectory (upfront planning)
  - error_recovery_rate: fraction of errors followed by a test run within 3 steps
  - read_before_first_edit: how many reads before the first edit (scoping breadth)
  - interleave_score: fraction of edits immediately followed by a test

Aggregates per-agent and prints a comparison table showing behavioral
zeitgeist shifts across model generations.

Usage: python behavioral_features.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE / "results"

AGENTS = {
    "Claude-3 Opus": RES / "fingerprints_claude3opus_n500.jsonl",
    "Claude-3.5 Sonnet": RES / "fingerprints_claude3.5sonnet_n500.jsonl",
    "Claude-3.7 (parent)": RES / "fingerprints_claude37_parent_n300.jsonl",
    "Claude-4 Sonnet": RES / "fingerprints_claude4sonnet_n500.jsonl",
    "GPT-4": RES / "fingerprints_gpt4_n500.jsonl",
    "GPT-4o": RES / "fingerprints_gpt4o_n500.jsonl",
    "SWE-agent-LM-32B": RES / "fingerprints_child_n500.jsonl",
    "DARS+R1": RES / "fingerprints_dars_r1_n300.jsonl",
}


def extract(atoms: list[str]) -> dict:
    non_think = [a for a in atoms if a != "think"]
    n = len(non_think)
    if n == 0:
        return {}

    cnt = Counter(non_think)

    search_first = int(non_think[0] == "search_repo") if non_think else 0

    reads_before_edit = 0
    for a in non_think:
        if a == "edit":
            break
        if a == "read_file":
            reads_before_edit += 1

    max_run = cur_run = 0
    for a in non_think:
        if a == "edit":
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    # think_early_frac uses the FULL sequence (think included), not non_think
    cutoff = max(1, len(atoms) // 4)
    early = atoms[:cutoff]
    think_early = early.count("think") / max(1, len(early))

    errors_followed = 0
    total_errors = 0
    for i, a in enumerate(non_think):
        if a == "error":
            total_errors += 1
            window = non_think[i + 1 : i + 4]
            if "run_test" in window:
                errors_followed += 1
    error_recovery = errors_followed / max(1, total_errors)

    # interleave_score = (edit, run_test) adjacent pairs / total edits
    edit_test_pairs = sum(
        1
        for i in range(len(non_think) - 1)
        if non_think[i] == "edit" and non_think[i + 1] == "run_test"
    )
    interleave_score = edit_test_pairs / max(1, cnt.get("edit", 0))

    return {
        "search_first": search_first,
        "reads_before_edit": reads_before_edit,
        "edit_run_max": max_run,
        "think_early_frac": think_early,
        "error_recovery_rate": error_recovery,
        "interleave_score": interleave_score,
        "n_actions": n,
    }


def run():
    agent_stats = {}

    for name, path in AGENTS.items():
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        all_feats = []
        for r in rows:
            atoms = r.get("atoms_canonical", [])
            f = extract(atoms)
            if f:
                f["resolved"] = r.get("resolved")
                all_feats.append(f)

        if not all_feats:
            continue

        keys = [
            "search_first",
            "reads_before_edit",
            "edit_run_max",
            "think_early_frac",
            "error_recovery_rate",
            "interleave_score",
            "n_actions",
        ]
        agg = {}
        for k in keys:
            vals = [f[k] for f in all_feats if f.get(k) is not None]
            agg[f"{k}_mean"] = float(np.mean(vals)) if vals else 0.0

        pass_sf = np.mean([f["search_first"] for f in all_feats if f.get("resolved") is True])
        fail_sf = np.mean([f["search_first"] for f in all_feats if f.get("resolved") is False])
        agg["search_first_pass"] = float(pass_sf) if not np.isnan(pass_sf) else 0.0
        agg["search_first_fail"] = float(fail_sf) if not np.isnan(fail_sf) else 0.0

        agent_stats[name] = agg

    FEATURES = [
        ("search_first_mean", "search-first %", "% trajectories starting with search"),
        ("reads_before_edit_mean", "reads before 1st edit", "scoping breadth before touching code"),
        ("edit_run_max_mean", "max edit streak", "longest batch of consecutive edits"),
        (
            "interleave_score_mean",
            "interleave score",
            "fraction of edits immediately followed by test",
        ),
        ("think_early_frac_mean", "upfront think frac", "fraction of think in first 25% of steps"),
        (
            "error_recovery_rate_mean",
            "error recovery rate",
            "fraction of errors followed by test in 3 steps",
        ),
        ("n_actions_mean", "avg trajectory length", "mean non-think actions"),
    ]

    print("=" * 90)
    print("BEHAVIORAL FEATURES BY AGENT  (exploration vs exploitation + zeitgeist)")
    print("=" * 90)

    agent_names = list(agent_stats.keys())
    col_w = 18
    header = f"  {'Feature':28s}" + "".join(f"  {n[:col_w]:>{col_w}s}" for n in agent_names)
    print(header)
    print("  " + "-" * (28 + (col_w + 2) * len(agent_names)))

    for col_key, label, description in FEATURES:
        row = f"  {label:28s}"
        vals = [agent_stats[n].get(col_key, 0.0) for n in agent_names]
        for v in vals:
            if col_key in (
                "search_first_mean",
                "interleave_score_mean",
                "think_early_frac_mean",
                "error_recovery_rate_mean",
            ):
                row += f"  {v:>{col_w}.1%}"
            else:
                row += f"  {v:>{col_w}.2f}"
        print(row)

    print()
    print("  search-first by outcome:")
    for name in agent_names:
        s = agent_stats[name]
        p = s.get("search_first_pass", 0)
        f = s.get("search_first_fail", 0)
        print(f"    {name:25s}  pass={p:.1%}  fail={f:.1%}  (Δ={p-f:+.1%})")

    out_path = RES / "behavioral_features_v1.json"
    out_path.write_text(json.dumps(agent_stats, indent=2))
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    run()
