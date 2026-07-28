"""Compare JSD against alternative divergence metrics on agent procedure distributions.

Holds the list→distribution mapping fixed (unigram/bigram count vectors
over canonical atoms) and varies only the metric. Reports Spearman rank
correlations between metric-induced pair orderings and which agent pairs
disagree most.

Usage: python metric_comparison.py
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

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

EPS = 1e-9  # Laplace smoothing constant


## Encodings


def to_unigram(atoms: list[str], vocab: list[str]) -> np.ndarray:
    cnt = Counter(atoms)
    v = np.array([cnt.get(a, 0) + EPS for a in vocab], dtype=float)
    return v / v.sum()


def to_bigram(atoms: list[str], vocab: list[str]) -> np.ndarray:
    pairs = [f"{atoms[i]}|{atoms[i+1]}" for i in range(len(atoms) - 1)]
    cnt = Counter(pairs)
    v = np.array([cnt.get(p, 0) + EPS for p in vocab], dtype=float)
    return v / v.sum()


def pool_distributions(rows, encoder, vocab):
    """Pool all trajectories of one agent into a single mixed distribution."""
    vecs = []
    for r in rows:
        atoms = r.get("atoms_canonical", [])
        if len(atoms) < 2:
            continue
        vecs.append(encoder(atoms, vocab))
    if not vecs:
        return np.ones(len(vocab)) / len(vocab)
    return np.mean(vecs, axis=0)


## Metrics


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2

    def _kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return (_kl(p, m) + _kl(q, m)) / 2


def kl_sym(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric KL = ½[KL(p||q) + KL(q||p)]."""
    return (float(np.sum(p * np.log(p / q))) + float(np.sum(q * np.log(q / p)))) / 2


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)) / 2)


def cosine_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(1 - np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q)))


METRICS = {
    "JSD": jsd,
    "KL-sym": kl_sym,
    "Hellinger": hellinger,
    "TotalVariation": total_variation,
    "Cosine": cosine_dist,
}


## Main


def run():
    agent_names = list(AGENTS.keys())
    corpora = {}
    for name, path in AGENTS.items():
        if not path.exists():
            print(f"  MISSING: {path.name}")
            continue
        corpora[name] = [json.loads(l) for l in path.read_text().splitlines()]
    agent_names = list(corpora.keys())
    pairs = list(itertools.combinations(agent_names, 2))

    # Build bigram vocab from observed transitions only (not all 81 pairs).
    # Unobserved pairs contribute only EPS and would dominate if kept in the
    # full 81-entry vocabulary.
    observed_bigrams: set[str] = set()
    for rows in corpora.values():
        for r in rows:
            atoms = r.get("atoms_canonical", [])
            for i in range(len(atoms) - 1):
                observed_bigrams.add(f"{atoms[i]}|{atoms[i+1]}")
    bigram_vocab = sorted(observed_bigrams)
    print(f"  Observed bigrams: {len(bigram_vocab)} (out of {len(CANON)**2} possible)")

    for encoding_name, encoder, vocab in [
        ("unigram", to_unigram, CANON),
        ("bigram", to_bigram, bigram_vocab),
    ]:
        print(f"\n{'='*68}")
        print(f"ENCODING: {encoding_name}  (vocab size={len(vocab)})")
        print(f"{'='*68}")

        dists = {n: pool_distributions(corpora[n], encoder, vocab) for n in agent_names}

        dist_vecs = {m: [] for m in METRICS}
        for a, b in pairs:
            for mname, mfn in METRICS.items():
                dist_vecs[mname].append(mfn(dists[a], dists[b]))

        print("\nSpearman r between metric-induced pair orderings:")
        mnames = list(METRICS.keys())
        print(f"  {'':18s}" + "".join(f"  {m:>14s}" for m in mnames))
        for m1 in mnames:
            row = f"  {m1:18s}"
            for m2 in mnames:
                if m1 == m2:
                    row += f"  {'1.000':>14s}"
                else:
                    r, _ = spearmanr(dist_vecs[m1], dist_vecs[m2])
                    row += f"  {r:>14.3f}"
            print(row)

        print("\nPair distances  (JSD vs Hellinger vs TotalVariation):")
        jsd_vals = dist_vecs["JSD"]
        hell_vals = dist_vecs["Hellinger"]
        tv_vals = dist_vecs["TotalVariation"]
        disagree = [
            (abs(jsd_vals[i] - hell_vals[i]), pairs[i], jsd_vals[i], hell_vals[i], tv_vals[i])
            for i in range(len(pairs))
        ]
        disagree.sort(reverse=True)
        print(f"  {'Pair':40s}  {'JSD':>6s}  {'Hell':>6s}  {'TV':>6s}")
        for _, (a, b), j, h, tv in disagree:
            print(f"  {a} vs {b:<28s}  {j:.3f}   {h:.3f}   {tv:.3f}")

    out = {}
    for encoding_name, encoder, vocab in [
        ("unigram", to_unigram, CANON),
        ("bigram", to_bigram, bigram_vocab),  # bigram_vocab built above from observations
    ]:
        dists = {n: pool_distributions(corpora[n], encoder, vocab) for n in agent_names}
        out[encoding_name] = {}
        for mname, mfn in METRICS.items():
            mat = {}
            for a, b in itertools.combinations(agent_names, 2):
                key = f"{a}||{b}"
                mat[key] = round(mfn(dists[a], dists[b]), 5)
            out[encoding_name][mname] = mat

    out_path = RES / "metric_comparison_v1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    run()
