"""Localize what is different across agents at the transition level.

For each pair of agents, find the bigrams (consecutive atom pairs) that
have the highest probability in one agent and lowest in the other.
These are the behavioral fault lines: the specific transitions that
fingerprint each model.

Also computes per-pair positional divergence curves so you can see
at which step a specific pair diverges, not just the global mean.

Usage: python discriminative_bigrams.py
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
    "Claude-4 Sonnet": RES / "fingerprints_claude4sonnet_n500.jsonl",
    "GPT-4": RES / "fingerprints_gpt4_n500.jsonl",
    "GPT-4o": RES / "fingerprints_gpt4o_n500.jsonl",
    "SWE-agent-LM-32B": RES / "fingerprints_child_n500.jsonl",
}

EPS = 1e-4  # additive smoothing; small but prevents log(0), doesn't swamp signal
MAX_K = 40
TOP_N = 6  # top discriminative bigrams per pair to print


def load_seqs(path: Path) -> list[list[str]]:
    return [json.loads(l).get("atoms_canonical", []) for l in path.read_text().splitlines()]


def bigram_dist(seqs: list[list[str]], vocab: list[str]) -> np.ndarray:
    cnt: Counter = Counter()
    for seq in seqs:
        for i in range(len(seq) - 1):
            cnt[f"{seq[i]}|{seq[i+1]}"] += 1
    v = np.array([cnt.get(b, 0) + EPS for b in vocab], dtype=float)
    return v / v.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    eps = 1e-12

    def kl(a, b):
        return float(np.sum(a * np.log((a + eps) / (b + eps))))

    return (kl(p, m) + kl(q, m)) / 2


def run():
    corpora = {n: load_seqs(p) for n, p in AGENTS.items() if p.exists()}
    agent_names = list(corpora.keys())

    vocab_set: set[str] = set()
    for seqs in corpora.values():
        for seq in seqs:
            for i in range(len(seq) - 1):
                vocab_set.add(f"{seq[i]}|{seq[i+1]}")
    vocab = sorted(vocab_set)

    dists = {n: bigram_dist(corpora[n], vocab) for n in agent_names}

    ## Discriminative bigrams per pair
    print("=" * 80)
    print("DISCRIMINATIVE BIGRAMS PER AGENT PAIR")
    print("  (transitions that most distinguish agent A from agent B)")
    print("=" * 80)

    FOCUS_PAIRS = [
        ("Claude-4 Sonnet", "GPT-4"),
        ("Claude-4 Sonnet", "Claude-3 Opus"),
        ("Claude-4 Sonnet", "SWE-agent-LM-32B"),
        ("Claude-3.5 Sonnet", "GPT-4o"),
        ("Claude-3 Opus", "Claude-3.5 Sonnet"),
    ]

    pair_bigrams = {}
    for a, b in FOCUS_PAIRS:
        if a not in dists or b not in dists:
            continue
        pa, pb = dists[a], dists[b]
        diff = pa - pb  # positive = more frequent in A
        ranked = sorted(zip(diff, vocab, strict=False), reverse=True)

        top_a = [(d, bg) for d, bg in ranked if d > 0][:TOP_N]  # A's signature
        top_b = [(d, bg) for d, bg in reversed(ranked) if d < 0][:TOP_N]  # B's signature

        print(f"\n{a}  vs  {b}  (bigram JSD={jsd(pa,pb):.3f})")
        print(f"  ── {a} signature (over-represented) ──")
        for d, bg in top_a:
            src, tgt = bg.split("|")
            bar = "█" * int(d * 400)
            print(f"    {src:12s} → {tgt:12s}  Δp={d:+.4f}  {bar}")
        print(f"  ── {b} signature (over-represented) ──")
        for d, bg in top_b:
            src, tgt = bg.split("|")
            bar = "█" * int(abs(d) * 400)
            print(f"    {src:12s} → {tgt:12s}  Δp={d:+.4f}  {bar}")

        pair_bigrams[f"{a}||{b}"] = {
            f"{a}_signature": [{"bigram": bg, "delta_p": round(d, 5)} for d, bg in top_a],
            f"{b}_signature": [{"bigram": bg, "delta_p": round(d, 5)} for d, bg in top_b],
        }

    ## Per-pair positional divergence curves
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

    def pos_dist(seqs, k, canon):
        cnt: Counter = Counter()
        n = 0
        for seq in seqs:
            if k < len(seq):
                cnt[seq[k]] += 1
                n += 1
        if n < 10:
            return None
        v = np.array([cnt.get(a, 0) + EPS for a in canon], dtype=float)
        return v / v.sum()

    print("\n" + "=" * 80)
    print("PER-PAIR POSITIONAL JSD CURVES  (which step diverges most for each pair)")
    print("=" * 80)

    pair_pos_curves = {}
    for a, b in FOCUS_PAIRS:
        if a not in corpora or b not in corpora:
            continue
        curve = []
        peak_k, peak_jsd = 0, 0.0
        for k in range(MAX_K):
            pa_k = pos_dist(corpora[a], k, CANON)
            pb_k = pos_dist(corpora[b], k, CANON)
            if pa_k is None or pb_k is None:
                break
            d = jsd(pa_k, pb_k)
            curve.append(d)
            if d > peak_jsd:
                peak_jsd = d
                peak_k = k

        print(f"\n  {a} vs {b}  — peak divergence at step {peak_k} (JSD={peak_jsd:.3f})")
        for k, d in enumerate(curve[:20]):
            bar = "█" * int(d * 40)
            print(f"    step {k:2d}  {d:.3f}  {bar}")
        if len(curve) > 20:
            print(f"    ... (steps 20–{len(curve)-1}: mean={np.mean(curve[20:]):.3f})")

        pair_pos_curves[f"{a}||{b}"] = {
            "curve": [round(v, 5) for v in curve],
            "peak_step": peak_k,
            "peak_jsd": round(peak_jsd, 5),
        }

    out = {"discriminative_bigrams": pair_bigrams, "positional_curves": pair_pos_curves}
    out_path = RES / "discriminative_bigrams_v1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    run()
