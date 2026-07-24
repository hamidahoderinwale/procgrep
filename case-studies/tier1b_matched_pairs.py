"""Tier 1B: same-outcome-different-procedure pair hunt.

For each pair of agents with labeled matched instances, find:
  - SODP (Same Outcome, Different Procedure): both pass or both fail,
    but procedural JSD is in the top quartile for that pair.
  - SODP examples are the empirical core of procgrep's thesis: outcome
    alone is insufficient; two agents can arrive at the same result
    via structurally distinct procedures.

Also finds:
  - DOSP (Different Outcome, Similar Procedure): one passes, one fails,
    but JSD is in the bottom quartile; procedure similarity didn't predict
    outcome alignment.

Outputs a JSON of top examples and prints a summary table.

Usage: python tier1b_matched_pairs.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RES = HERE / "results"

AGENTS = {
    "Claude-3.5 Sonnet": RES / "fingerprints_claude3.5sonnet_n500.jsonl",
    "Claude-4 Sonnet": RES / "fingerprints_claude4sonnet_n500.jsonl",
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


def load(path: Path) -> dict[str, dict]:
    return {json.loads(l)["instance_id"]: json.loads(l) for l in path.read_text().splitlines()}


def dist(atoms: list[str]) -> np.ndarray:
    cnt = Counter(atoms)
    v = np.array([cnt.get(a, 0) + EPS for a in CANON], dtype=float)
    return v / v.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return (kl(p, m) + kl(q, m)) / 2


def sequence_similarity(a: list[str], b: list[str]) -> float:
    """Longest common subsequence ratio as a sequence-level similarity."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return 2 * lcs / (m + n)


def analyse_pair(name_a: str, data_a: dict, name_b: str, data_b: dict):
    common = set(data_a) & set(data_b)
    labeled = [
        iid
        for iid in common
        if data_a[iid].get("resolved") is not None and data_b[iid].get("resolved") is not None
    ]
    if len(labeled) < 20:
        return None

    records = []
    for iid in labeled:
        ra, rb = data_a[iid], data_b[iid]
        atoms_a = ra.get("atoms_canonical", [])
        atoms_b = rb.get("atoms_canonical", [])
        pa = dist(atoms_a)
        pb = dist(atoms_b)
        j = jsd(pa, pb)
        lcs = sequence_similarity(atoms_a, atoms_b)
        res_a = bool(ra["resolved"])
        res_b = bool(rb["resolved"])
        records.append(
            {
                "instance_id": iid,
                "resolved_a": res_a,
                "resolved_b": res_b,
                "same_outcome": res_a == res_b,
                "jsd": j,
                "lcs_sim": lcs,
                "len_a": len(atoms_a),
                "len_b": len(atoms_b),
                "atoms_a": atoms_a[:20],  # first 20 for vignette
                "atoms_b": atoms_b[:20],
            }
        )

    jsds = [r["jsd"] for r in records]
    q75 = np.percentile(jsds, 75)
    q25 = np.percentile(jsds, 25)

    sodp = [r for r in records if r["same_outcome"] and r["jsd"] >= q75]
    dosp = [r for r in records if not r["same_outcome"] and r["jsd"] <= q25]

    sodp.sort(key=lambda x: -x["jsd"])
    dosp.sort(key=lambda x: x["jsd"])

    return {
        "pair": f"{name_a} vs {name_b}",
        "n_matched": len(labeled),
        "n_both_pass": sum(1 for r in records if r["resolved_a"] and r["resolved_b"]),
        "n_both_fail": sum(1 for r in records if not r["resolved_a"] and not r["resolved_b"]),
        "n_a_only": sum(1 for r in records if r["resolved_a"] and not r["resolved_b"]),
        "n_b_only": sum(1 for r in records if not r["resolved_a"] and r["resolved_b"]),
        "mean_jsd": float(np.mean(jsds)),
        "q25_jsd": float(q25),
        "q75_jsd": float(q75),
        "n_sodp": len(sodp),
        "n_dosp": len(dosp),
        "top_sodp": sodp[:5],
        "top_dosp": dosp[:3],
    }


def print_results(results: list[dict]):
    print("=" * 80)
    print("TIER 1B: SAME-OUTCOME, DIFFERENT-PROCEDURE PAIRS")
    print("=" * 80)
    for res in results:
        if res is None:
            continue
        print(f"\n{res['pair']}  (n={res['n_matched']} matched)")
        print(
            f"  outcomes:  both-pass={res['n_both_pass']}  "
            f"both-fail={res['n_both_fail']}  "
            f"A-only={res['n_a_only']}  B-only={res['n_b_only']}"
        )
        print(
            f"  JSD:  mean={res['mean_jsd']:.3f}  "
            f"q25={res['q25_jsd']:.3f}  q75={res['q75_jsd']:.3f}"
        )
        print(f"  SODP (same outcome, top-quartile JSD): {res['n_sodp']}")
        print(f"  DOSP (diff outcome, bot-quartile JSD): {res['n_dosp']}")

        if res["top_sodp"]:
            print("\n  Top SODP examples (highest procedure divergence, same outcome):")
            for ex in res["top_sodp"][:3]:
                outcome = "PASS" if ex["resolved_a"] else "FAIL"
                print(f"    [{outcome}] {ex['instance_id']}")
                print(
                    f"      JSD={ex['jsd']:.3f}  LCS={ex['lcs_sim']:.2f}  "
                    f"len_a={ex['len_a']} len_b={ex['len_b']}"
                )
                print(f"      A: {ex['atoms_a'][:10]}")
                print(f"      B: {ex['atoms_b'][:10]}")


def run():
    corpora = {n: load(p) for n, p in AGENTS.items() if p.exists()}
    names = list(corpora.keys())

    FOCUS = [
        ("Claude-3.5 Sonnet", "SWE-agent-LM-32B"),
        ("Claude-4 Sonnet", "SWE-agent-LM-32B"),
        ("Claude-4 Sonnet", "Claude-3.5 Sonnet"),
        ("Claude-4 Sonnet", "GPT-4o"),
        ("GPT-4o", "SWE-agent-LM-32B"),
    ]

    all_results = []
    for na, nb in FOCUS:
        if na not in corpora or nb not in corpora:
            continue
        res = analyse_pair(na, corpora[na], nb, corpora[nb])
        all_results.append(res)

    print_results(all_results)

    out = [r for r in all_results if r is not None]
    # Remove atoms from saved output to keep file small
    for r in out:
        for ex in r.get("top_sodp", []) + r.get("top_dosp", []):
            ex.pop("atoms_a", None)
            ex.pop("atoms_b", None)

    out_path = RES / "tier1b_matched_pairs_v1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    run()
