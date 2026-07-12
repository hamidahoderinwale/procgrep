"""Gate verdict for a controls run: does enforcement take, and do controls behave?

Reads the three paired run dirs (process_enforce_hard, ablate_run_test,
null_control) under a gate directory and prints, per spec: guard activity,
per-instance action counts (baseline vs enforced), and the baseline<->enforced
population JSD under one shared vocabulary. The three baseline arms are
same-condition replicates, so their pairwise JSDs are the empirical noise
floor the enforcement movement must clear. Optionally folds in swebench
resolve reports (--grades-dir with <arm>.<spec>.json files of resolved ids).

    uv run python scripts/gate_report.py runs/gate_20260711
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from procgrep_runner.measure import arm_traces

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.jsd import jsd

SPECS = ("process_enforce_hard", "ablate_run_test", "null_control")


def population_dist(traces, vocab):
    totals = np.zeros(vocab.size, dtype=np.float64)
    for fp in encode(traces, vocab=vocab):
        totals += np.asarray(fp.counts, dtype=np.float64)
    return totals / totals.sum()


def action_counts(run_dir: Path, arm: str) -> dict[str, int]:
    out = {}
    for path in sorted((run_dir / "arms" / arm).glob("*.traj.json")):
        record = json.loads(path.read_text())
        iid = record.get("instance_id") or path.name.split(".r")[0]
        out[iid] = (record.get("procgrep_runner") or {}).get("action_count")
    return out


def guard_totals(run_dir: Path) -> dict[str, int]:
    blocked = steered = checks = 0
    for path in sorted((run_dir / "arms" / "enforced").glob("*.traj.json")):
        g = (json.loads(path.read_text()).get("procgrep_runner") or {}).get("guard") or {}
        blocked += g.get("blocked", 0)
        steered += g.get("steered", 0)
        checks += g.get("checks", 0)
    return {"checks": checks, "blocked": blocked, "steered": steered}


def main(gate_dir: Path) -> None:
    runs = {s: gate_dir / s for s in SPECS if (gate_dir / s).exists()}
    arms = {
        (s, arm): arm_traces(d, arm) for s, d in runs.items() for arm in ("baseline", "enforced")
    }
    vocab = fit_bpe([t.atoms for traces in arms.values() for t in traces], vocab_size=64, seed=0)
    dists = {k: population_dist(v, vocab) for k, v in arms.items() if v}

    print(f"gate: {gate_dir}  |  shared vocab size {vocab.size}\n")
    floor = [
        round(jsd(dists[(a, "baseline")], dists[(b, "baseline")]), 4)
        for a, b in combinations(runs, 2)
        if (a, "baseline") in dists and (b, "baseline") in dists
    ]
    print(f"baseline<->baseline noise floor (same-condition JSD): {floor}\n")

    for spec, run_dir in runs.items():
        move = jsd(dists[(spec, "baseline")], dists[(spec, "enforced")])
        guard = guard_totals(run_dir)
        base_actions = action_counts(run_dir, "baseline")
        enf_actions = action_counts(run_dir, "enforced")
        print(f"== {spec}")
        print(f"   guard: {guard}")
        print(f"   baseline<->enforced JSD: {move:.4f}")
        for iid in sorted(base_actions):
            print(f"   {iid}: baseline {base_actions[iid]} -> enforced {enf_actions.get(iid)}")
        exits = [
            (t.metadata.get("instance_id"), t.metadata.get("exit_status"))
            for t in arms[(spec, "enforced")]
            if t.metadata.get("exit_status") != "Submitted"
        ]
        if exits:
            print(f"   non-submitted enforced runs: {exits}")
        print()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
