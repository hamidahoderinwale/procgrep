"""Actions per resolved task — step-budget efficiency, 9 agents.

Total canonical actions across all 300 attempts divided by tasks resolved.
Horizontal bar, sorted by efficiency (fewest actions per resolution).

Reads:
    ~/learning-from-dev/bidirect-align-dev-traces/output/paper2_pilot/aggregate_metrics_extended.json
    ~/learning-from-dev/bidirect-align-dev-traces/output/paper2_pilot/extended_pass_fail.json
Writes:
    ~/learning-from-dev/procgrep/docs/figures/fig_steps_per_resolved.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figtheme import init, style_axes, BLUE, COPPER, GREEN, OLIVE, MAGENTA, INK

init()

BIDIRECT = Path(__file__).resolve().parents[2] / "bidirect-align-dev-traces"
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "fig_steps_per_resolved.png"

SUBMISSION_LABEL = {
    "20240402_sweagent_claude3opus":                          "Claude-3",
    "20240402_sweagent_gpt4":                                 "GPT-4",
    "20240620_sweagent_claude3.5sonnet":                      "Claude-3.5",
    "20240728_sweagent_gpt4o":                                "GPT-4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219":           "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514":             "Claude-4",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1":      "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022":      "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3":                          "Moatless+V3",
}

AGENT_COLORS = {
    "Claude-3":              COPPER,
    "Claude-3.5":            GREEN,
    "Claude-3.7-thinking":   "#20856A",
    "Claude-4":              "#187860",
    "GPT-4":                 BLUE,
    "GPT-4o":                MAGENTA,
    "DARS+R1":               "#7A1240",
    "Agentless+Claude-3.5":  "#3A6DB5",
    "Moatless+V3":           OLIVE,
}

LITE_TOTAL = 300


def main() -> None:
    agg = json.loads((BIDIRECT / "output" / "paper2_pilot" / "aggregate_metrics_extended.json").read_text())
    pf  = json.loads((BIDIRECT / "output" / "paper2_pilot" / "extended_pass_fail.json").read_text())

    rows = []
    for sub_id, agent in SUBMISSION_LABEL.items():
        info = pf.get(sub_id, {})
        n_resolved = len(set(info.get("resolved", [])))
        m = agg["metrics"].get(agent, {})
        mean_len = m.get("canonical_length_mean") or m.get("mean_canonical_length") or m.get("mean_atoms")
        if mean_len is None or n_resolved == 0:
            continue
        rows.append({
            "agent": agent,
            "steps_per_resolved": mean_len * LITE_TOTAL / n_resolved,
        })

    rows.sort(key=lambda r: r["steps_per_resolved"])
    agents = [r["agent"] for r in rows]
    values = [r["steps_per_resolved"] for r in rows]
    colors = [AGENT_COLORS[a] for a in agents]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    y_pos = range(len(agents))
    bars = ax.barh(list(y_pos), values, color=colors, height=0.6)

    for i, val in enumerate(values):
        ax.text(val + 2, i, f"{val:.0f}", va="center", fontsize=8.5, color=INK)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(agents, fontsize=9)
    ax.set_xlabel("Actions per resolved task", fontsize=10)
    ax.set_title("Actions per resolved task", fontsize=11)
    ax.set_xlim(0, max(values) * 1.12)

    style_axes(ax)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
