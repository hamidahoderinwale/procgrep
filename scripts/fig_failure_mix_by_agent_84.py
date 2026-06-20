"""Pass rate vs composition failure rate for 84 SWE-bench Lite agents.

Each point is one agent. x = overall pass rate (resolved / 300).
y = composition failure rate (fraction of failures that are novel_composition).
Color and connecting line by model family (Claude, GPT, Open-weight).

Reads:
    ~/learning-from-dev/bidirect-align-dev-traces/output/compositional_generalization/agent_libraries.json
    ~/learning-from-dev/bidirect-align-dev-traces/output/compositional_generalization/instance_classification.json
Writes:
    ~/learning-from-dev/procgrep/docs/figures/failure_mix_by_agent_84.png
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figtheme import BLUE, COPPER, GREEN, init, style_axes

init()

CG = Path(__file__).resolve().parents[2] / "bidirect-align-dev-traces" / "output" / "compositional_generalization"
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "failure_mix_by_agent_84.png"

TOTAL = 300

CLAUDE_TOKENS = ("claude", "sonnet", "opus", "haiku")
GPT_TOKENS    = ("gpt", "openai", "o3", "o1")

FAMILY_COLOR = {
    "Claude":      COPPER,
    "GPT":         BLUE,
    "Open-weight": GREEN,
}


def agent_family(submission_id: str) -> str:
    s = submission_id.lower()
    if any(t in s for t in CLAUDE_TOKENS):
        return "Claude"
    if any(t in s for t in GPT_TOKENS):
        return "GPT"
    return "Open-weight"


def main() -> None:
    al = json.loads((CG / "agent_libraries.json").read_text())
    ic = json.loads((CG / "instance_classification.json").read_text())

    agents = list(al.keys())

    # Accumulate failure type counts per agent
    agent_counts: dict[str, Counter] = {a: Counter() for a in agents}
    for _iid, agent_classes in ic.items():
        for agent, cls in agent_classes.items():
            if agent in agent_counts:
                agent_counts[agent][cls] += 1

    rows = []
    for agent in agents:
        n_solved = al[agent]["n_solved"]
        pass_rate = n_solved / TOTAL
        fc = agent_counts[agent]
        total_fails = sum(fc.values())
        comp_rate = fc["novel_composition"] / total_fails if total_fails > 0 else 0.0
        rows.append({
            "agent":     agent,
            "family":    agent_family(agent),
            "pass_rate": pass_rate,
            "comp_rate": comp_rate,
        })

    # Sort within each family by pass_rate for the line
    from collections import defaultdict
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    for fam in by_family:
        by_family[fam].sort(key=lambda r: r["pass_rate"])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # Background scatter — all agents, muted
    all_x = [r["pass_rate"] for r in rows]
    all_y = [r["comp_rate"] for r in rows]
    ax.scatter(all_x, all_y, color="#C0BDB8", s=22, zorder=1, alpha=0.6)

    # Per-family line + highlighted scatter
    for fam, color in FAMILY_COLOR.items():
        frows = by_family[fam]
        xs = [r["pass_rate"] for r in frows]
        ys = [r["comp_rate"] for r in frows]
        ax.plot(xs, ys, color=color, linewidth=1.4, zorder=2, alpha=0.85)
        ax.scatter(xs, ys, color=color, s=36, zorder=3, label=fam)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlabel("Pass rate", fontsize=10)
    ax.set_ylabel("Composition failure rate", fontsize=10)
    ax.set_title("Pass rate and composition failure rate", fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
