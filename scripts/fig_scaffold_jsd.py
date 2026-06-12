"""Intent: generate the scaffold-vs-model figure for the essay, a log-axis dot
plot of action-mix Jensen-Shannon divergence between trace groups. It shows the
headline decomposition: within one scaffold the model barely moves the
fingerprint, while crossing scaffolds moves it one to two orders of magnitude
more. Reads the local spine store and writes docs/figures/scaffold_jsd.svg.

Design decisions (benefit / price):
1. Compute JSD from the store at build time, do not hardcode the numbers.
   Benefit: the figure regenerates and stays honest if the store changes.
   Price: needs the local parquet present to rebuild.
2. Hand-emit a small SVG in the editorial palette rather than a plotting lib.
   Benefit: crisp, dependency-light, matches the explorer's inline charts.
   Price: a little layout math here instead of a declarative spec.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pandas as pd

STORE = Path("data/procgrep_spines.parquet")
OUT = Path("docs/figures/scaffold_jsd.svg")
INK, OLIVE, RULE, COPPER, TEAL = "#14110E", "#585E53", "#d9d4cc", "#CB4D20", "#20A380"


def _mix(rows: list[str]) -> dict[str, float]:
    c: Counter[str] = Counter(a for sp in rows for a in sp.split())
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def _jsd(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def kl(a: dict[str, float]) -> float:
        return sum(a.get(k, 0.0) * math.log2(a[k] / m[k]) for k in a if a.get(k, 0.0) > 0)

    return 0.5 * kl(p) + 0.5 * kl(q)


def main() -> int:
    df = pd.read_parquet(STORE)

    def mix_where(dataset: str, agent: str | None = None) -> dict[str, float]:
        sub = df[df.dataset == dataset]
        if agent:
            sub = sub[sub.agent == agent]
        return _mix(list(sub.spine))

    ne = "nebius/SWE-agent-trajectories"
    rows = [
        (
            "Llama 8B vs 70B, one scaffold",
            _jsd(mix_where(ne, "swe-agent-llama-8b"), mix_where(ne, "swe-agent-llama-70b")),
            "within",
        ),
        (
            "Llama 8B vs 405B, one scaffold",
            _jsd(mix_where(ne, "swe-agent-llama-8b"), mix_where(ne, "swe-agent-llama-405b")),
            "within",
        ),
        (
            "OpenHands vs OpenHands variant",
            _jsd(
                mix_where("nebius/SWE-rebench-openhands-trajectories"),
                mix_where("nvidia/SWE-Zero-openhands-trajectories"),
            ),
            "across",
        ),
        (
            "SWE-agent vs OpenHands",
            _jsd(mix_where(ne), mix_where("nebius/SWE-rebench-openhands-trajectories")),
            "across",
        ),
        (
            "SWE-agent vs OpenHands variant",
            _jsd(mix_where(ne), mix_where("nvidia/SWE-Zero-openhands-trajectories")),
            "across",
        ),
    ]

    w, h, ml, mr, mt, mb = 640, 250, 250, 70, 18, 44
    pw, ph = w - ml - mr, h - mt - mb
    xmin, xmax = math.log10(8e-4), math.log10(0.5)

    def x(v: float) -> float:
        return ml + (math.log10(max(v, 8e-4)) - xmin) / (xmax - xmin) * pw

    svg = [
        f'<svg viewBox="0 0 {w} {h}" class="lqplot" role="img" aria-label="scaffold versus model divergence">'
    ]
    for v, lab in [(1e-3, "0.001"), (1e-2, "0.01"), (1e-1, "0.1")]:
        xx = x(v)
        svg.append(f'<line x1="{xx:.1f}" y1="{mt}" x2="{xx:.1f}" y2="{mt + ph}" stroke="#ece7df"/>')
        svg.append(
            f'<text x="{xx:.1f}" y="{mt + ph + 16}" text-anchor="middle" class="ax">{lab}</text>'
        )
    svg.append(
        f'<text x="{ml + pw}" y="{mt + ph + 33}" text-anchor="end" class="ax">action-mix JSD, log scale</text>'
    )
    n = len(rows)
    for i, (label, v, grp) in enumerate(rows):
        y = mt + (i + 0.5) / n * ph
        color = TEAL if grp == "within" else COPPER
        svg.append(
            f'<text x="{ml - 10}" y="{y + 3:.1f}" text-anchor="end" class="ax" fill="{INK}">{label}</text>'
        )
        svg.append(
            f'<line x1="{x(8e-4):.1f}" y1="{y:.1f}" x2="{x(v):.1f}" y2="{y:.1f}" stroke="{RULE}"/>'
        )
        svg.append(
            f'<circle cx="{x(v):.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{label}: JSD {v:.4f}</title></circle>'
        )
        svg.append(
            f'<text x="{x(v) + 9:.1f}" y="{y + 3:.1f}" class="ax" fill="{color}">{v:.3f}</text>'
        )
    svg.append("</svg>")
    OUT.write_text("\n".join(svg))
    print(f"wrote {OUT}  ({[round(v, 4) for _, v, _ in rows]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
