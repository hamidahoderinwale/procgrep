"""OOD (out-of-distribution) behavioral analysis.

Two OOD axes, both computable from existing fingerprints (no re-fetch):

A. INTRA-AGENT OOD SCORE
   For each trajectory T belonging to agent A, compute JSD between T's
   per-atom distribution and A's mean distribution. High score = this
   trajectory looks anomalous for this agent.
   Question: do failures concentrate in the high-OOD tail?

B. CROSS-AGENT OOD SCORE
   For each trajectory T from agent A, compute JSD to every OTHER agent's
   mean distribution. The nearest other agent = the model whose style T
   most resembles. Useful for detecting style convergence / transfer.
   Question: when the child's trajectory is "OOD for the child," which
   parent-era agent does it most resemble?

Both scores are computed at both canonical and native alphabet levels.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROCGREP_SRC = Path(__file__).resolve().parent.parent / "procgrep" / "src"
if _PROCGREP_SRC.exists() and str(_PROCGREP_SRC) not in sys.path:
    sys.path.insert(0, str(_PROCGREP_SRC))

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT_DIR = RESULTS / "ood_analysis_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_FILES = [
    ("Claude-3 Opus", "fingerprints_claude3opus.jsonl"),
    ("Claude-3.5 Sonnet", "fingerprints_claude3.5sonnet.jsonl"),
    ("Claude-4 Sonnet", "fingerprints_claude4sonnet.jsonl"),
    ("SWE-agent-LM-32B", "fingerprints_child_n500.jsonl"),
    ("GPT-4", "fingerprints_gpt4.jsonl"),
    ("GPT-4o", "fingerprints_gpt4o.jsonl"),
]


def load(layer: str = "canonical") -> dict[str, list[dict]]:
    """Load all agents' trace dicts keyed by agent name."""
    result: dict[str, list[dict]] = {}
    for name, fname in AGENT_FILES:
        path = RESULTS / fname
        if not path.exists():
            continue
        rows = []
        for line in path.read_text().splitlines():
            d = json.loads(line)
            if d.get(f"atoms_{layer}"):
                rows.append(d)
        result[name] = rows
    return result


def atom_freq(atoms: list[str]) -> Counter:
    return Counter(a for a in atoms if a != "think")


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability vectors."""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = lambda x, y: np.sum(np.where(x > 0, x * np.log2(x / np.where(y > 0, y, 1e-300)), 0))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def to_vec(freq: Counter, vocab: list[str]) -> np.ndarray:
    total = sum(freq.values())
    if total == 0:
        return np.zeros(len(vocab))
    v = np.array([freq.get(a, 0) / total for a in vocab], dtype=float)
    return v


def main() -> None:
    for layer in ("canonical", "native"):
        print(f"=== OOD analysis @ {layer} ===")
        data = load(layer)
        agents = list(data.keys())
        if not agents:
            print("  no data loaded")
            continue

        # Build global vocabulary across all agents
        all_atoms: set[str] = set()
        for rows in data.values():
            for r in rows:
                all_atoms.update(a for a in r[f"atoms_{layer}"] if a != "think")
        vocab = sorted(all_atoms)

        # Per-agent mean frequency vector
        agent_mean: dict[str, np.ndarray] = {}
        for agent, rows in data.items():
            combined: Counter = Counter()
            for r in rows:
                combined.update(atom_freq(r[f"atoms_{layer}"]))
            agent_mean[agent] = to_vec(combined, vocab)

        # --- A. Intra-agent OOD scores ---
        print("  A. Intra-agent OOD (JSD to own mean), resolve-stratified:")
        intra_results: list[dict] = []
        for agent, rows in data.items():
            mean_vec = agent_mean[agent]
            ood_scores = []
            for r in rows:
                traj_vec = to_vec(atom_freq(r[f"atoms_{layer}"]), vocab)
                score = jsd(traj_vec, mean_vec)
                resolved = r.get("resolved")
                ood_scores.append(
                    {"score": score, "resolved": resolved, "instance_id": r["instance_id"]}
                )
            passed = [x["score"] for x in ood_scores if x["resolved"] is True]
            failed = [x["score"] for x in ood_scores if x["resolved"] is False]
            all_scores = [x["score"] for x in ood_scores]
            median_all = float(np.median(all_scores)) if all_scores else 0
            median_pass = float(np.median(passed)) if passed else 0
            median_fail = float(np.median(failed)) if failed else 0
            print(
                f"    {agent:24s}  median OOD={median_all:.3f}  pass={median_pass:.3f}  fail={median_fail:.3f}"
            )
            for item in ood_scores:
                item["agent"] = agent
                item["layer"] = layer
            intra_results.extend(ood_scores)

        # --- B. Cross-agent OOD: nearest neighbor in procedure-space ---
        print("  B. Cross-agent nearest neighbor (% assigned to each agent):")
        nn_table: dict[str, Counter] = {a: Counter() for a in agents}
        for agent, rows in data.items():
            for r in rows:
                traj_vec = to_vec(atom_freq(r[f"atoms_{layer}"]), vocab)
                best_agent = min(
                    agents,
                    key=lambda a: jsd(traj_vec, agent_mean[a]) if a != agent else float("inf"),
                )
                nn_table[agent][best_agent] += 1

        for agent in agents:
            total = sum(nn_table[agent].values())
            nn_str = ", ".join(
                f"{a[:16]}:{100 * n / total:.0f}%"
                for a, n in nn_table[agent].most_common(3)
                if total > 0
            )
            print(f"    {agent:24s}  top nearest neighbors → {nn_str}")

        # --- Plot: intra-agent OOD score distribution by pass/fail ---
        fig, axes = plt.subplots(1, len(agents), figsize=(3 * len(agents), 4), sharey=False)
        for ax, agent in zip(axes, agents, strict=False):
            pass_scores = [
                x["score"] for x in intra_results if x["agent"] == agent and x["resolved"] is True
            ]
            fail_scores = [
                x["score"] for x in intra_results if x["agent"] == agent and x["resolved"] is False
            ]
            if pass_scores:
                ax.hist(pass_scores, bins=10, alpha=0.6, color="#2ca02c", label="pass")
            if fail_scores:
                ax.hist(fail_scores, bins=10, alpha=0.6, color="#d62728", label="fail")
            ax.set_title(agent[:16], fontsize=9)
            ax.set_xlabel("OOD score")
            ax.legend(fontsize=7)
        plt.suptitle(f"Intra-agent OOD score distribution ({layer})", y=1.01)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"ood_distribution_{layer}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: ood_distribution_{layer}.png")

        # Save intra-agent results
        out_path = OUT_DIR / f"ood_scores_{layer}.jsonl"
        with out_path.open("w") as f:
            for item in intra_results:
                f.write(json.dumps(item) + "\n")
        print()


if __name__ == "__main__":
    main()
