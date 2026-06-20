"""Regenerated figure: BPE vs PrefixSpan V-measure comparison.

What it shows: horizontal bar chart comparing V-measure (vs agent labels) for
BPE at K=64 and K=128 against PrefixSpan at K=64 — three vocabulary induction
conditions. Values are hardcoded from the paper evaluation run.

Style: procgrep figtheme (monospace, Tufte-minimal, no grid).
Output: ~/learning-from-dev/procgrep/docs/figures/fig_bpe_vs_prefixspan.png
"""

import sys

sys.path.insert(0, "/Users/hamidaho/learning-from-dev/procgrep/scripts")

import matplotlib.pyplot as plt
import numpy as np
from figtheme import BLUE, COPPER, INK, RULE, init, style_axes

init()

# Hardcoded V-measure values from paper evaluation
methods = ["PrefixSpan K=64", "BPE K=64", "BPE K=128"]
values  = [0.505,              0.606,       0.626      ]
colors  = [COPPER,             BLUE,        BLUE       ]

fig, ax = plt.subplots(figsize=(6.2, 3.0))

y_pos = np.arange(len(methods))
bars = ax.barh(y_pos, values, color=colors, height=0.55)

# Reference line at PrefixSpan value to show gain
ax.axvline(values[0], color=RULE, linewidth=1.0, linestyle="--", zorder=0)

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=9)
ax.set_xlabel("V-measure vs agent labels", fontsize=10)
ax.set_xlim(0, 0.75)
ax.set_title("Vocabulary induction: BPE vs PrefixSpan", fontsize=11)

# Value labels on bars
for bar, val in zip(bars, values, strict=False):
    ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9,
            color=INK, fontfamily="monospace")

style_axes(ax)

fig.savefig(
    "/Users/hamidaho/learning-from-dev/procgrep/docs/figures/fig_bpe_vs_prefixspan.png",
    dpi=200, facecolor="white", bbox_inches="tight"
)
print("wrote fig_bpe_vs_prefixspan.png")
