"""Shared figure theme for procgrep custom plots — the project palette + a
Tufte-minimal axes styler, so every regenerated figure matches one contract
(palette is a project-level decision, not a per-figure one).

Usage:
    from figtheme import init, style_axes, BLUE, COPPER, OLIVE, GREEN, MAGENTA
    init()
    fig, ax = plt.subplots(...)
    ...
    style_axes(ax); fig.savefig(..., dpi=200, facecolor="white")
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Project palette (matches docs/assets/css/theme.css).
BLUE = "#5692E5"
COPPER = "#CB4D20"
GREEN = "#20A380"
OLIVE = "#585E53"
MAGENTA = "#B4184F"
RULE = "#d9d4cc"
INK = "#14110E"
PAPER = "#F7F5F2"
# Default categorical order, muted, warm-cool balanced, colorblind-aware.
PALETTE = [BLUE, COPPER, GREEN, MAGENTA, OLIVE]


def init() -> None:
    """Global rcParams: monospace, no grid (data-ink only)."""
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


def style_axes(ax) -> None:
    """Tufte baseline: drop top/right spines, mute the rest, color labels."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=OLIVE, labelsize=9)
    ax.title.set_color(INK)
    ax.title.set_size(11)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
