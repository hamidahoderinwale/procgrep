"""Regenerated figure: pass rate vs files changed in submitted patch, per agent.

What it shows: line chart showing how pass rate changes as the number of files
in the submitted patch increases (bins: 1, 2, 3+), with one line per agent.
Agents are drawn in procgrep palette colors (BLUE family = GPT; COPPER family =
Claude; GREEN = DARS+R1). Dashed reference line at y=0 suppressed; no
parentheticals in labels.

Data: ~/learning-from-dev/bidirect-align-dev-traces/output/paper2_pilot/extended_pass_fail.json
      ~/learning-from-dev/bidirect-align-dev-traces/output/trajectories/.cache/<agent>/

Run from repo root or pass absolute paths — this script uses absolute paths.

Style: procgrep figtheme.
Output: ~/learning-from-dev/procgrep/docs/figures/fig_patch_files_passrate.png
"""

import glob
import json
import os
import re
import sys

sys.path.insert(0, "/Users/hamidaho/learning-from-dev/procgrep/scripts")

from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figtheme import init, style_axes

init()

CACHE = Path(
    "/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces/output/trajectories/.cache"
)
with open(
    "/Users/hamidaho/learning-from-dev/bidirect-align-dev-traces/output/paper2_pilot/extended_pass_fail.json"
) as fh:
    pf = json.load(fh)

NAME = {
    "20240402_sweagent_claude3opus": "Claude-3",
    "20240620_sweagent_claude3.5sonnet": "Claude-3.5",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "Claude-3.7-thinking",
    "20250526_sweagent_claude-4-sonnet-20250514": "Claude-4",
    "20240402_sweagent_gpt4": "GPT-4",
    "20240728_sweagent_gpt4o": "GPT-4o",
    "20250205_dars_agent_claude_3.5_sonnet_deepseek_r1": "DARS+R1",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022": "Agentless+Claude-3.5",
    "20250111_moatless_deepseek_v3": "Moatless+V3",
}

# figtheme palette assignment: Claude family = COPPER shades, GPT = BLUE shades, DARS = GREEN
FAM_COLORS = {
    "Claude-3": "#CB8050",  # COPPER-light
    "Claude-3.5": "#CB4D20",  # COPPER (canonical)
    "Claude-3.7-thinking": "#8B3010",  # COPPER-dark
    "Claude-4": "#5B1808",  # COPPER-darkest
    "GPT-4": "#7BB4F0",  # BLUE-light
    "GPT-4o": "#5692E5",  # BLUE (canonical)
    "DARS+R1": "#20A380",  # GREEN
}

GIT = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.M)
PLUS = re.compile(r"^\+\+\+ b/(\S+)", re.M)


def patch_files(diff: str) -> int:
    if not diff or not diff.strip():
        return 0
    files = {m.group(2) for m in GIT.finditer(diff)}
    if not files:
        files = {m.group(1) for m in PLUS.finditer(diff)}
    return len(files)


def binof(n):
    if n <= 0:
        return None
    return "1" if n == 1 else ("2" if n == 2 else "3+")


rows = []
for key, agent in NAME.items():
    d = CACHE / key
    if not d.exists():
        print(f"MISSING dir {key}")
        continue
    resolved = set(pf.get(key, {}).get("resolved", []))
    cell = defaultdict(lambda: [0, 0])
    for fp in glob.glob(str(d / "*.json")):
        iid = os.path.basename(fp)[:-5]
        try:
            with open(fp) as fh:
                o = json.load(fh)
        except Exception:
            continue
        sub = o.get("info", {}).get("submission", "") or ""
        if not sub.strip() and isinstance(o.get("content"), dict):
            sub = o["content"].get("info", {}).get("submission", "") or ""
        if not sub.strip() and isinstance(o.get("content"), list):
            for m in reversed(o["content"]):
                t = str(m.get("content", "")) if isinstance(m, dict) else str(m)
                if "diff --git" in t:
                    sub = t
                    break
        if not sub.strip():
            sub = o.get("submission", "") or ""
        nf = patch_files(sub)
        b = binof(nf)
        if b is None:
            continue
        ok = iid in resolved
        cell[b][1] += 1
        cell[b][0] += int(ok)
    for b in ("1", "2", "3+"):
        p, t = cell[b]
        if t >= 5:
            rows.append({"agent": agent, "bin": b, "pass_rate": 100 * p / t, "n": t})

df = pd.DataFrame(rows)
df = df[df.agent.isin(FAM_COLORS)]

order = ["1", "2", "3+"]

fig, ax = plt.subplots(figsize=(6.2, 4.4))

legend_handles = []
for agent, color in FAM_COLORS.items():
    sub = df[df.agent == agent].copy()
    if sub.empty:
        continue
    sub = sub.set_index("bin").reindex(order)
    x_vals = range(len(order))
    y_vals = sub["pass_rate"].values
    # only plot bins with data
    mask = ~np.isnan(y_vals.astype(float))
    xp = [i for i, m in enumerate(mask) if m]
    yp = [y_vals[i] for i in xp]
    ax.plot(
        xp,
        yp,
        color=color,
        linewidth=1.8,
        marker="o",
        markersize=5,
        markerfacecolor=color,
        markeredgewidth=0,
    )
    legend_handles.append(
        mlines.Line2D([], [], color=color, linewidth=1.8, marker="o", markersize=5, label=agent)
    )

ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.set_xlabel("Files in submitted patch", fontsize=10)
ax.set_ylabel("Pass rate", fontsize=10)
ax.set_ylim(0, 70)
ax.set_title("Pass rate by number of files changed", fontsize=11)

legend = ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="upper right", ncol=1)

style_axes(ax)

fig.savefig(
    "/Users/hamidaho/learning-from-dev/procgrep/docs/figures/fig_patch_files_passrate.png",
    dpi=200,
    facecolor="white",
    bbox_inches="tight",
)
print("wrote fig_patch_files_passrate.png")
