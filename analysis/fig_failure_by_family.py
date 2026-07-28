"""Failure mode ECDF by model family — 3 panels.

For each failure category (novel composition / novel primitive / familiar pattern),
plots the empirical CDF of within-agent failure rates across model families
(Claude, GPT-4, Open-weight).

Within-agent failure rate for a category = fraction of an agent's classified
failures that fall in that category. Model family assigned by backbone name
in submission ID.

Reads:
    ~/learning-from-dev/bidirect-align-dev-traces/output/compositional_generalization/agent_libraries.json
    ~/learning-from-dev/bidirect-align-dev-traces/output/compositional_generalization/instance_classification.json
Writes:
    ~/learning-from-dev/procgrep/docs/figures/failure_by_family.png
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figtheme import BLUE, COPPER, GREEN, INK, init, style_axes

init()

CG = (
    Path(__file__).resolve().parents[2]
    / "bidirect-align-dev-traces"
    / "output"
    / "compositional_generalization"
)
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "failure_by_family.png"

CLAUDE_TOKENS = ("claude", "sonnet", "opus", "haiku")
GPT_TOKENS = ("gpt", "openai", "o3", "o1")

FAMILY_COLOR = {
    "Claude": COPPER,
    "GPT-4": BLUE,
    "Open-weight": GREEN,
}

PANEL_LABEL = {
    "novel_composition": "Novel composition",
    "novel_primitive": "Novel primitive",
    "familiar": "Familiar pattern",
}


def agent_family(submission_id: str) -> str:
    s = submission_id.lower()
    if any(t in s for t in CLAUDE_TOKENS):
        return "Claude"
    if any(t in s for t in GPT_TOKENS):
        return "GPT-4"
    return "Open-weight"


def ecdf(values: list[float]):
    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    # prepend 0 for a clean step from the origin
    xs = np.concatenate([[xs[0]], xs])
    ys = np.concatenate([[0.0], ys])
    return xs, ys


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

    # Per-agent: within-failure fractions per category
    cat_order = ["novel_composition", "novel_primitive", "familiar"]
    family_rates: dict[str, dict[str, list[float]]] = {
        fam: {cat: [] for cat in cat_order} for fam in FAMILY_COLOR
    }

    for agent in agents:
        fam = agent_family(agent)
        fc = agent_counts[agent]
        total = sum(fc.values())
        if total == 0:
            continue
        for cat in cat_order:
            rate = fc[cat] / total
            family_rates[fam][cat].append(rate)

    # Debug counts
    for fam in FAMILY_COLOR:
        for cat in cat_order:
            n = len(family_rates[fam][cat])
            print(f"{fam} / {cat}: n={n}")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)

    for ax, cat in zip(axes, cat_order, strict=False):
        for fam, color in FAMILY_COLOR.items():
            rates = family_rates[fam][cat]
            if not rates:
                continue
            xs, ys = ecdf(rates)
            ax.step(xs, ys, where="post", color=color, linewidth=1.8, label=fam)

        ax.set_title(PANEL_LABEL[cat], fontsize=10, color=INK)
        ax.set_xlabel("Within-agent failure rate", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        style_axes(ax)

    axes[0].set_ylabel("Cumulative proportion of agents", fontsize=9)

    # Single legend below, all panels
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=2, label=fam)
        for fam, color in FAMILY_COLOR.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.tight_layout(rect=[0, 0.0, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
