"""Step share vs token share by stage and agent — dumbbell chart.

Data: bidirect-align-dev-traces/output/paper2_pilot/bpe_sequences_extended.jsonl
      bidirect-align-dev-traces/output/paper2_pilot/step_resources.json

For each agent panel: a MAGENTA dot at step fraction and a BLUE dot at token
fraction per stage, connected by a thin line. Gap shows where step count
overstates or understates token cost. Agents without token data show step
dots only.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/hamidaho/learning-from-dev/procgrep/scripts")
from figtheme import BLUE, INK, MAGENTA, RULE, init, style_axes

ROOT = Path("/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces")

init()

STAGES = ["Explore", "Browse", "Edit", "Test", "Shell", "Finish"]
THRESHOLD = 0.010

FAMILY_ORDER = [
    "Claude-3",
    "Claude-3.5",
    "Claude-3.7-thinking",
    "Claude-4",
    "GPT-4",
    "GPT-4o",
    "DARS+R1",
    "Agentless+Claude-3.5",
    "Moatless+V3",
]


def classify(atom: str) -> str:
    if atom.startswith("SEARCH"):
        return "Explore"
    if atom.startswith(("OPEN", "NAV", "FIND")):
        return "Browse"
    if atom.startswith(("EDIT", "CREATE")):
        return "Edit"
    if atom.startswith("RUN"):
        return "Test"
    if atom.startswith("SHELL_"):
        return "Shell"
    if atom.startswith("SUBMIT"):
        return "Finish"
    return "Other"


atoms_data = json.loads((ROOT / "output/paper2_pilot/step_resources.json").read_text())["atoms"]

agent_stage_tok: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
agent_total_tok: dict[str, float] = defaultdict(float)

for atom, info in atoms_data.items():
    s = classify(atom)
    for ag, cnt in info.get("by_agent", {}).items():
        tok = cnt * info["mean_tokens_per_use"]
        agent_stage_tok[ag][s] += tok
        agent_total_tok[ag] += tok

step_counts: dict[str, Counter] = defaultdict(Counter)
total_steps: dict[str, int] = defaultdict(int)

seq_path = ROOT / "output/paper2_pilot/bpe_sequences_extended.jsonl"
with seq_path.open() as f:
    for line in f:
        r = json.loads(line)
        ag, atoms = r["agent"], r["canonical"]
        n = max(len(atoms), 1)
        total_steps[ag] += n
        for a in atoms:
            step_counts[ag][classify(a)] += 1

agents_in_data = [a for a in FAMILY_ORDER if a in total_steps]
n_agents = len(agents_in_data)

ncols = 3
nrows = int(np.ceil(n_agents / ncols))

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(6.2, 2.1 * nrows + 0.6),
    sharex=True,
)
axes_flat = axes.flatten()

for i, agent in enumerate(agents_in_data):
    ax = axes_flat[i]
    n = total_steps[agent]
    t = agent_total_tok.get(agent, 0)
    has_tokens = t > 0

    visible_stages = []
    for s in STAGES:
        sf = step_counts[agent].get(s, 0) / n
        tf = agent_stage_tok[agent].get(s, 0) / t if has_tokens else None
        tok_val = tf if tf is not None else 0.0
        if sf >= THRESHOLD or tok_val >= THRESHOLD:
            visible_stages.append((s, sf, tf))

    y_positions = list(range(len(visible_stages)))
    stage_labels = [s for s, _, _ in visible_stages]

    for j, (_s, sf, tf) in enumerate(visible_stages):
        if sf >= THRESHOLD and tf is not None and tf >= THRESHOLD:
            ax.plot([sf, tf], [j, j], color=RULE, linewidth=1.2, zorder=1)
        if sf >= THRESHOLD:
            ax.scatter([sf], [j], color=MAGENTA, s=28, zorder=3, linewidths=0)
        if tf is not None and tf >= THRESHOLD:
            ax.scatter([tf], [j], color=BLUE, s=28, zorder=3, linewidths=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(stage_labels, fontsize=8)
    ax.set_xlim(0, 0.68)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_title(agent, fontsize=8, pad=3)

    style_axes(ax)

    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

for i in range(n_agents, len(axes_flat)):
    axes_flat[i].set_visible(False)

legend_handles = [
    mpatches.Patch(color=MAGENTA, label="Steps"),
    mpatches.Patch(color=BLUE, label="Tokens"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, 0.0),
)

fig.suptitle(
    "Step share vs token share, by stage and agent",
    fontsize=10,
    x=0.05,
    ha="left",
    color=INK,
)

fig.tight_layout(rect=[0, 0.05, 1, 0.97])

fig.savefig(
    "/Users/hamidaho/learning-from-dev/procgrep/docs/figures/fig_step_vs_token_cost.png",
    dpi=200,
    facecolor="white",
    bbox_inches="tight",
)
plt.close(fig)
print("wrote fig_step_vs_token_cost.png")
