"""Intent: figure asset for the representation-comparison table in the essay, a
horizontal bar chart of retrieval F1 by representation, colored by source. Shows
the headline: structural/deterministic representations top the table, narrative
descriptions sit below, and the LLM divergence classifier ranges wildly. The
numbers are the table's; this only visualizes them. Writes
docs/figures/representation_f1.svg.

Design decisions (benefit / price):
1. The values are passed in literally from the paper's table, not recomputed.
   Benefit: the figure matches the table exactly and stays the author's data.
   Price: if the table changes, update ROWS here too.
2. Point values draw as bars; the two ranges (LLM classifier, random baseline)
   draw as light bands with their span.
   Benefit: one frame shows both the ranking and the LLM's instability.
   Price: mixed mark types need a short legend, kept minimal.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("docs/figures/representation_f1.svg")
INK, OLIVE, RULE, COPPER, GRAY = "#14110E", "#585E53", "#d9d4cc", "#CB4D20", "#b7b1a7"

# (label, value-or-(lo,hi), source-group). Values verbatim from the paper table.
ROWS = [
    ("Structural pattern overlap", 0.347, "deterministic"),
    ("Action sequence distance", 0.274, "deterministic"),
    ("Edit action overlap", 0.256, "deterministic"),
    ("Narrative description", 0.177, "narrative"),
    ("Agent plan description", 0.155, "narrative"),
    ("Divergence classifier", (0.000, 0.363), "llm"),
    ("Random retrieval", (0.13, 0.24), "baseline"),
]
FILL = {"deterministic": COPPER, "narrative": OLIVE, "llm": GRAY, "baseline": RULE}


def main() -> int:
    w, h, ml, mr, mt, mb = 660, 320, 210, 60, 18, 56
    pw, ph = w - ml - mr, h - mt - mb
    xmax = 0.4

    def x(v: float) -> float:
        return ml + (v / xmax) * pw

    svg = [
        f'<svg viewBox="0 0 {w} {h}" class="lqplot" role="img" aria-label="retrieval F1 by representation">'
    ]
    for tick in (0.0, 0.1, 0.2, 0.3, 0.4):
        xx = x(tick)
        svg.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt + ph}" stroke="#ece7df"/>')
        svg.append(
            f'<text x="{xx:.1f}" y="{mt + ph + 16}" text-anchor="middle" class="ax">{tick:.1f}</text>'
        )
    svg.append(
        f'<text x="{ml + pw}" y="{mt + ph + 34}" text-anchor="end" class="ax">retrieval F1 at k = 1, n = 289</text>'
    )

    n = len(ROWS)
    bh = ph / n * 0.55
    for i, (label, val, grp) in enumerate(ROWS):
        cy = mt + (i + 0.5) / n * ph
        color = FILL[grp]
        svg.append(
            f'<text x="{ml - 10}" y="{cy + 3:.1f}" text-anchor="end" class="ax" fill="{INK}">{label}</text>'
        )
        if isinstance(val, tuple):
            lo, hi = val
            svg.append(
                f'<rect x="{x(lo):.1f}" y="{cy - bh / 2:.1f}" width="{x(hi) - x(lo):.1f}" height="{bh:.1f}" '
                f'fill="{color}" opacity="0.5"><title>{label}: F1 {lo:.3f} to {hi:.3f}</title></rect>'
            )
            svg.append(
                f'<text x="{x(hi) + 8:.1f}" y="{cy + 3:.1f}" class="ax" fill="{OLIVE}">{lo:.2f} to {hi:.2f}</text>'
            )
        else:
            svg.append(
                f'<rect x="{ml}" y="{cy - bh / 2:.1f}" width="{x(val) - ml:.1f}" height="{bh:.1f}" '
                f'fill="{color}"><title>{label}: F1 {val:.3f}</title></rect>'
            )
            svg.append(
                f'<text x="{x(val) + 8:.1f}" y="{cy + 3:.1f}" class="ax" fill="{color}">{val:.3f}</text>'
            )

    legend = [("deterministic", COPPER), ("varies by model", OLIVE), ("LLM, unstable", GRAY)]
    lx = ml
    for lab, col in legend:
        svg.append(f'<rect x="{lx}" y="{h - 16}" width="10" height="10" fill="{col}"/>')
        svg.append(f'<text x="{lx + 15}" y="{h - 7}" class="ax">{lab}</text>')
        lx += 150
    svg.append("</svg>")
    OUT.write_text("\n".join(svg))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
