"""Shared figure theme for procgrep custom plots — palette + Tufte-minimal axes.

Usage (LIVE / essay figures — monospace):
    from figtheme import init, style_axes, save_fig, BLUE
    init()
    fig, ax = plt.subplots(...)
    style_axes(ax)
    save_fig(fig, "out.png")

Usage (PLATEAU figures — sans-serif):
    from figstyle import init, style_axes, save_fig, BLUE   # plateau/figstyle.py
    init()
"""

from __future__ import annotations

from pathlib import Path

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

_SANS_STACK = ["Arial", "Helvetica", "DejaVu Sans"]
_DPI = 200


def init(*, sans: bool = False) -> None:
    """Global rcParams. Pass ``sans=True`` for PLATEAU figures."""
    if sans:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = _SANS_STACK
        plt.rcParams["mathtext.fontset"] = "dejavusans"
    else:
        plt.rcParams["font.family"] = "monospace"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.dpi"] = _DPI


def style_axes(
    ax,
    *,
    hide_left: bool = False,
    hide_bottom: bool = False,
    bottom_bounds: tuple[float, float] | None = None,
) -> None:
    """Tufte baseline: drop top/right spines, mute the rest, color labels."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_color(RULE)
    if hide_bottom:
        ax.spines["bottom"].set_visible(False)
    else:
        ax.spines["bottom"].set_color(RULE)
        if bottom_bounds is not None:
            ax.spines["bottom"].set_bounds(*bottom_bounds)
    ax.tick_params(colors=OLIVE, labelsize=9, length=3)
    ax.title.set_color(INK)
    ax.title.set_size(11)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)


def style_title(ax, text: str, *, loc: str = "center", pad: float = 10) -> None:
    """Consistent title placement (``loc='left'`` for PLATEAU bar/dot panels)."""
    ax.set_title(text, color=INK, size=11, loc=loc, pad=pad)


def style_axes_ccdf(ax) -> None:
    """Log-log CCDF panels (cascade size, file degree)."""
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)


def style_axes_barh(ax, *, hide_left: bool = True) -> None:
    """Horizontal bar / dumbbell panels with y-category labels."""
    style_axes(ax, hide_left=hide_left)
    ax.tick_params(axis="y", length=0)


def style_axes_heatmap(ax) -> None:
    """Matrix heatmaps (procedure JSD, etc.)."""
    for sp in ax.spines.values():
        sp.set_color(RULE)
    ax.tick_params(colors=OLIVE, labelsize=9)
    ax.title.set_color(INK)
    ax.title.set_size(11)


def save_fig(fig, path: str | Path, *, dpi: int = _DPI, tight: bool = True) -> Path:
    """Write PNG with project defaults."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": dpi, "facecolor": "white"}
    if tight:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(path, **kwargs)
    return path
