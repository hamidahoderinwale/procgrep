"""The induced procedure vocabulary as a tree: merges decomposed down to atoms.

Renders the saved pooled BPE vocabulary and its deepest procedures. Each merge glues two
existing tokens, so every procedure is a binary tree whose leaves are atoms:
reading the leaves top-to-bottom spells the procedure. Dots at the root mark
which agents emit it.

This is the LEARNED vocabulary, not a hand-declared taxonomy: the groupings come
out of BPE over real sessions, and the procedures drawn are every one at the
maximum merge depth, so nothing is hand-picked.

Corpus: SALT-NLP/SWE-chat transcripts, 25 sessions per interface.
Style: Instrument Sans through the project figtheme.
The original code and pinned dataset were used once to recover this exact tree;
rendering now reads the tracked derived vocabulary without network access.
Output: docs/figures/induced_vocab_tree/induced_vocab_tree.png
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from figtheme import BLUE, COPPER, GREEN, INK, MAGENTA, OLIVE, RULE, init, save_fig
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ["Claude Code", "OpenCode", "Codex", "Gemini CLI"]
COLORS = dict(zip(AGENTS, [BLUE, COPPER, GREEN, MAGENTA], strict=True))
SEP = "▁"
INPUT = ROOT / "docs/paper/data/induced_vocab_tree.json"
OUT = ROOT / "docs/figures/induced_vocab_tree/induced_vocab_tree.png"


## layout


def layout(tok, kids, y0):
    """Place a binary merge tree: leaves stacked from y0 down, atoms at x=0.

    Returns (node, next_y). A node carries its own children rather than being
    re-matched by name later -- the two children of a merge are often the same
    token (edit+edit), so a name lookup cannot tell the subtrees apart.
    """
    if tok not in kids:
        return {"tok": tok, "x": 0.0, "y": y0, "kids": []}, y0 + 1.0
    left, right = kids[tok]
    ln, y1 = layout(left, kids, y0)
    rn, y2 = layout(right, kids, y1)
    node = {
        "tok": tok,
        "x": max(ln["x"], rn["x"]) + 1.0,
        "y": (ln["y"] + rn["y"]) / 2,
        "kids": [ln, rn],
    }
    return node, y2


def walk(node):
    yield node
    for kid in node["kids"]:
        yield from walk(kid)


def main():
    data = json.loads(INPUT.read_text())
    kids = {left + SEP + right: (left, right) for left, right in data["merges"]}
    depth = dict.fromkeys(data["atoms"], 0)
    for tok, (left, right) in kids.items():
        depth[tok] = 1 + max(depth[left], depth[right])
    deepest = max(depth[t] for t in kids)
    shown = sorted((t for t in kids if depth[t] == deepest), key=lambda t: len(t.split(SEP)))
    assert shown == data["shown"]
    users = data["users"]

    init()
    font = Path(font_manager.findfont("Instrument Sans", fallback_to_default=False)).resolve()
    expected = (ROOT / "docs/assets/fonts/InstrumentSans.ttf").resolve()
    if font != expected:
        raise RuntimeError("Instrument Sans did not resolve to the bundled font")
    roots, y = [], 0.0
    for tok in shown:
        root, y = layout(tok, kids, y)
        roots.append(root)
        y += 1.4  # gap between procedures
    ymax = y - 1.4  # drop the trailing gap
    span = max(n["x"] for r in roots for n in walk(r))

    fig, ax = plt.subplots(figsize=(6.6, max(4.0, 0.24 * ymax)))

    def ypos(v):  # first procedure at the top
        return ymax - v

    def xpos(xd):  # root at the left, atom leaves at the right
        return span - xd

    dots_x = [xpos(span) - 0.95 + i * 0.22 for i in range(len(AGENTS))]

    for root in roots:
        for node in walk(root):
            if not node["kids"]:
                ax.text(
                    xpos(0) + 0.12,
                    ypos(node["y"]),
                    node["tok"],
                    ha="left",
                    va="center",
                    fontsize=9.5,
                    color=OLIVE,
                )
                continue
            for kid in node["kids"]:
                ax.plot(
                    [xpos(node["x"]), xpos(node["x"]), xpos(kid["x"])],
                    [ypos(node["y"]), ypos(kid["y"]), ypos(kid["y"])],
                    color=RULE,
                    lw=0.9,
                    solid_joinstyle="miter",
                )
        ax.scatter(xpos(root["x"]), ypos(root["y"]), s=14, color=INK, zorder=5)
        # which agents emit this procedure
        for x, agent in zip(dots_x, AGENTS, strict=True):
            on = agent in users.get(root["tok"], ())
            ax.scatter(
                x,
                ypos(root["y"]),
                s=20,
                zorder=5,
                color=COLORS[agent] if on else "#d3cec6",
                edgecolors="none",
            )

    for x, agent in zip(dots_x, AGENTS, strict=True):
        ax.text(
            x,
            ypos(-0.5),
            agent,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=COLORS[agent],
            rotation=55,
            rotation_mode="anchor",
        )
    ax.text(
        dots_x[0],
        ypos(ymax) - 0.1,
        "colored = the agent emits this procedure",
        ha="left",
        va="center",
        fontsize=8.5,
        color=OLIVE,
    )

    ax.set_xlim(dots_x[0] - 0.2, xpos(0) + 1.35)
    ax.set_ylim(ypos(ymax) - 0.5, ypos(0) + 1.4)
    ax.axis("off")
    fig.suptitle(
        "Deepest procedures in the learned vocabulary",
        x=0.03,
        ha="left",
        fontsize=12,
        fontweight="normal",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"saved {save_fig(fig, OUT)}")
    meta = {
        "artifact": OUT.name,
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "input": "../../paper/data/induced_vocab_tree.json",
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "generator": "analysis/fig_induced_vocab_tree.py",
        "font": "Instrument Sans",
        "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
        "profile": "Personal data-viz face and palette requested for the web essay",
        "renderer": "Matplotlib retained for the established binary-tree layout",
        "analysis_rerun": False,
        "historical_recovery": data["_meta"],
        "review_status": "independent visual review pending",
    }
    OUT.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
