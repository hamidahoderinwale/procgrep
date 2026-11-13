"""Render the essay figures whose original plotted values are saved in the repo.

The JSON records the source hashes and observation units. This entry point only
renders those measurements; it does not fit models or reconstruct a cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from figtheme import BLUE, COPPER, GREEN, INK, OLIVE, PAPER, RULE, init, save_fig, style_axes
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "docs/paper/data/figure_inputs.json"
FAMILIES = {"Claude": (COPPER, "o"), "GPT": (BLUE, "s"), "Open-weight": (GREEN, "^")}
NAMES = (
    "failure_mix_by_agent_84",
    "failure_by_family",
    "fig_interface_jsd_matrix",
    "fig_bpe_vs_prefixspan",
)


def scatter(data: dict):
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    for family, (color, marker) in FAMILIES.items():
        rows = [r for r in data["failures"] if r["family"] == family]
        ax.scatter(
            [r["resolved"] / r["attempts"] for r in rows],
            [r["failure_counts"]["novel_composition"] / r["classified_failures"] for r in rows],
            color=color,
            marker=marker,
            s=42,
            alpha=1,
            linewidths=0.5,
            edgecolors="white",
            clip_on=False,
            label=family,
        )
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Tasks resolved (%)")
    ax.set_ylabel("Novel composition among classified failures (%)")
    ax.set_title("Pass rates and composition failures", loc="left", pad=14)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    return fig


def distributions(data: dict):
    categories = {
        "novel_composition": "Novel composition",
        "novel_primitive": "Novel primitive",
        "familiar": "Familiar pattern",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1), sharey=True, sharex=True)
    for ax, (category, label) in zip(axes, categories.items(), strict=True):
        for y, (family, (color, marker)) in enumerate(FAMILIES.items()):
            rows = sorted(
                (r for r in data["failures"] if r["family"] == family), key=lambda r: r["agent"]
            )
            values = [r["failure_counts"][category] / r["classified_failures"] for r in rows]
            # Deterministic offsets expose overlap; vertical position has no numeric meaning.
            offsets = [((i % 7) - 3) * 0.055 for i in range(len(rows))]
            ax.scatter(
                values,
                np.asarray(offsets) + y,
                color=color,
                marker=marker,
                s=28,
                alpha=1,
                clip_on=False,
            )
        ax.set(xlim=(0, 1), ylim=(2.45, -0.45))
        ax.set_yticks(range(len(FAMILIES)), list(FAMILIES))
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Share of classified failures (%)")
        ax.set_title(label, loc="left", pad=12)
        style_axes(ax, hide_left=True)
        ax.tick_params(axis="y", length=0)
    fig.suptitle("Failure composition across model families", x=0.06, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def matrix(data: dict):
    mat = data["interface_jsd"]
    labels = list(mat)
    values = np.array([[mat[a][b] for b in labels] for a in labels])
    assert values.shape == (4, 4)
    assert np.allclose(values, values.T)
    assert np.allclose(np.diag(values), 0)
    assert np.all((values >= 0) & (values <= 1))
    fig, ax = plt.subplots(figsize=(6.9, 5.4))
    cmap = LinearSegmentedColormap.from_list("procgrep_jsd", [PAPER, "#91c9c1", "#16544b"])
    im = ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11, color=INK)
    ax.set_xticks(range(4), labels, rotation=25, ha="right")
    ax.set_yticks(range(4), labels)
    ax.tick_params(length=0, labelsize=10, colors=OLIVE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Procedural divergence across coding interfaces", loc="left", pad=16, fontsize=13)
    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    bar.set_label("Jensen-Shannon divergence", fontsize=10)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=9)
    return fig


def induction(data: dict):
    rows = data["induction"]
    labels = [f"{r['method']}   K = {r['vocabulary_size']}" for r in rows]
    values = [r["vmeasure"] for r in rows]
    colors = [COPPER if r["method"] == "PrefixSpan" else BLUE for r in rows]
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    for y, (value, color) in enumerate(zip(values, colors, strict=True)):
        ax.plot([0, value], [y, y], color=RULE, linewidth=1.3)
        ax.scatter(value, y, color=color, s=55, zorder=3)
        ax.text(value + 0.025, y, f"{value:.3f}", va="center", fontsize=11, color=INK)
    ax.set(xlim=(0, 1), ylim=(2.5, -0.5))
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlabel("V-measure")
    ax.set_title("Vocabulary agreement with agent identity", loc="left", pad=14, fontsize=13)
    style_axes(ax, hide_left=True)
    ax.tick_params(axis="y", length=0)
    return fig


RENDERERS = dict(zip(NAMES, [scatter, distributions, matrix, induction], strict=True))


def render_one(name: str) -> Path:
    data = json.loads(INPUT.read_text())
    init()
    plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.titleweight": "normal"})
    resolved = Path(font_manager.findfont("Instrument Sans", fallback_to_default=False)).resolve()
    expected = (ROOT / "docs/assets/fonts/InstrumentSans.ttf").resolve()
    if resolved != expected:
        raise RuntimeError(f"Instrument Sans resolved outside the bundled font: {resolved}")
    fig = RENDERERS[name](data)
    if name != "failure_by_family":
        fig.tight_layout()
    target = ROOT / "docs/figures" / name / f"{name}.png"
    save_fig(fig, target)
    plt.close(fig)
    meta = {
        "artifact": target.name,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "input": "../../paper/data/figure_inputs.json",
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "generator": "analysis/render_essay_figures.py",
        "font": "Instrument Sans",
        "font_sha256": hashlib.sha256(expected.read_bytes()).hexdigest(),
        "profile": "Personal data-viz face and palette requested for the web essay",
        "renderer": "Matplotlib retained for the established static figure pipeline",
        "analysis_rerun": False,
        "review_status": "local render completed; independent visual review pending",
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2) + "\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", choices=NAMES)
    args = parser.parse_args()
    for name in args.figures or NAMES:
        print(render_one(name))


if __name__ == "__main__":
    main()
