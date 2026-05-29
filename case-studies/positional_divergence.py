"""Positional divergence: which step in a trajectory diverges most across agents?

For each absolute position k, build the distribution of canonical atoms that
appear at step k across all trajectories of each agent. Compute cross-agent
JSD at each k and find the peak = "the step that is most different."

Also computes relative-position (percentile-binned) profiles so agents with
different average trajectory lengths are still comparable.

Usage: python positional_divergence.py
"""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE / "results"

AGENTS = {
    "Claude-3 Opus": RES / "fingerprints_claude3opus_n500.jsonl",
    "Claude-3.5 Sonnet": RES / "fingerprints_claude3.5sonnet_n500.jsonl",
    "Claude-4 Sonnet": RES / "fingerprints_claude4sonnet_n500.jsonl",
    "GPT-4": RES / "fingerprints_gpt4_n500.jsonl",
    "GPT-4o": RES / "fingerprints_gpt4o_n500.jsonl",
    "SWE-agent-LM-32B": RES / "fingerprints_child_n500.jsonl",
}

CANON = [
    "edit",
    "read_file",
    "run_test",
    "search_repo",
    "create_file",
    "delete_file",
    "think",
    "error",
    "other",
]
EPS = 1e-9
N_REL = 10  # relative position bins (deciles)
MAX_K = 40  # absolute positions to analyse


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2

    def _kl(a, b):
        return float(np.sum(a * np.log((a + EPS) / (b + EPS))))

    return (_kl(p, m) + _kl(q, m)) / 2


def make_dist(counts: Counter, vocab: list[str]) -> np.ndarray:
    v = np.array([counts.get(a, 0) + EPS for a in vocab], dtype=float)
    return v / v.sum()


def run():
    corpora: dict[str, list[list[str]]] = {}
    for name, path in AGENTS.items():
        if not path.exists():
            continue
        seqs = []
        for l in path.read_text().splitlines():
            r = json.loads(l)
            atoms = r.get("atoms_canonical", [])
            if atoms:
                seqs.append(atoms)
        corpora[name] = seqs

    agent_names = list(corpora.keys())

    # ── Absolute position profiles ────────────────────────────────────────────
    # For each agent and position k, distribution over atoms at position k.
    abs_profiles: dict[str, dict[int, np.ndarray]] = {}
    for name, seqs in corpora.items():
        pos_counts: dict[int, Counter] = defaultdict(Counter)
        for seq in seqs:
            for k, atom in enumerate(seq[:MAX_K]):
                pos_counts[k][atom] += 1
        abs_profiles[name] = {
            k: make_dist(cnt, CANON)
            for k, cnt in pos_counts.items()
            if sum(cnt.values()) >= 10  # at least 10 trajectories reach this depth
        }

    # Cross-agent mean JSD at each position
    print("=" * 68)
    print("ABSOLUTE POSITION DIVERGENCE  (mean pairwise JSD across all agent pairs)")
    print(f"{'Step k':>8s}  {'Mean JSD':>10s}  {'Most-divergent pair'}")
    print("=" * 68)

    all_positions = sorted(set(k for p in abs_profiles.values() for k in p))
    abs_divergence = {}
    for k in all_positions:
        present = [(n, abs_profiles[n][k]) for n in agent_names if k in abs_profiles[n]]
        if len(present) < 2:
            continue
        jsds = []
        max_jsd = -1
        max_pair = ("", "")
        for (na, pa), (nb, pb) in itertools.combinations(present, 2):
            d = jsd(pa, pb)
            jsds.append(d)
            if d > max_jsd:
                max_jsd = d
                max_pair = (na, nb)
        mean_jsd = np.mean(jsds)
        abs_divergence[k] = mean_jsd
        bar = "█" * int(mean_jsd * 30)
        print(f"  step {k:2d}  {mean_jsd:.3f}  {bar}  [{max_pair[0]} vs {max_pair[1]}]")

    peak_k = max(abs_divergence, key=abs_divergence.get)
    print(f"\n  Peak divergence at step {peak_k}  (JSD={abs_divergence[peak_k]:.3f})")

    # Which atom drives the divergence at the peak step?
    print(f"\n  Atom distribution at peak step k={peak_k}:")
    print(f"  {'Agent':25s}" + "".join(f"  {a[:8]:>8s}" for a in CANON))
    for name in agent_names:
        if peak_k not in abs_profiles[name]:
            continue
        dist = abs_profiles[name][peak_k]
        row = f"  {name:25s}"
        for v in dist:
            row += f"  {v:8.3f}"
        print(row)

    # ── Relative position profiles ────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("RELATIVE POSITION DIVERGENCE  (trajectory deciles 0%–100%)")
    print("=" * 68)

    rel_profiles: dict[str, dict[int, np.ndarray]] = {}
    for name, seqs in corpora.items():
        bin_counts: dict[int, Counter] = defaultdict(Counter)
        for seq in seqs:
            n = len(seq)
            if n == 0:
                continue
            for i, atom in enumerate(seq):
                bucket = min(int(i / n * N_REL), N_REL - 1)
                bin_counts[bucket][atom] += 1
        rel_profiles[name] = {b: make_dist(cnt, CANON) for b, cnt in bin_counts.items()}

    print(f"{'Decile':>8s}  {'Mean JSD':>10s}  {'Most-divergent pair'}")
    rel_divergence = {}
    for b in range(N_REL):
        present = [(n, rel_profiles[n][b]) for n in agent_names if b in rel_profiles[n]]
        if len(present) < 2:
            continue
        jsds = []
        max_jsd = -1
        max_pair = ("", "")
        for (na, pa), (nb, pb) in itertools.combinations(present, 2):
            d = jsd(pa, pb)
            jsds.append(d)
            if d > max_jsd:
                max_jsd = d
                max_pair = (na, nb)
        mean_jsd = np.mean(jsds)
        rel_divergence[b] = mean_jsd
        label = f"{b*10}–{(b+1)*10}%"
        bar = "█" * int(mean_jsd * 30)
        print(f"  {label:>8s}  {mean_jsd:.3f}  {bar}  [{max_pair[0]} vs {max_pair[1]}]")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "absolute": {str(k): v for k, v in abs_divergence.items()},
        "relative": {str(b): v for b, v in rel_divergence.items()},
        "peak_absolute_step": int(peak_k),
    }
    out_path = RES / "positional_divergence_v1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    run()
