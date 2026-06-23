"""Intent: render the cross-interface procedural-divergence matrix for the essay,
the interface counterpart to fig_jsd_matrix_full_canonical (which grades agents).
Adds interface as a fourth fingerprint axis alongside lineage, family, and
scaffold: pairwise action-mix JSD between four coding interfaces.

Design decisions (benefit / price):
1. Read the matrix from plateau/results.json (cross_interface_bpe) rather than
   recompute, since it is the validated asset that feeds the plateau paper.
   Benefit: one source of truth across both documents. Price: regenerating the
   divergence numbers is the plateau pipeline's job, not this figure's.
2. Match the canonical agent matrix exactly: same teal sequential ramp, mono
   labels, top-left title, JSD colorbar. Benefit: the two matrices read as one
   family. Price: the ramp is hardcoded to that figure, not figtheme's palette.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figtheme import INK, OLIVE, init  # noqa: E402

RESULTS = ROOT / "plateau" / "results.json"
OUT = ROOT / "docs" / "figures" / "fig_interface_jsd_matrix.png"
OUT_JSON = ROOT / "docs" / "paper" / "data" / "interface_jsd.json"
OUT_HTML = ROOT / "docs" / "figures" / "interface_jsd.html"
D3CHARTS = ROOT / "docs" / "explorer" / "d3charts.js"

# Self-contained interactive page mirroring fig_scaffold_jsd: D3 from the CDN,
# the shared module inlined, the matrix rendered with hover-for-exact-value.
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cross-interface procedural divergence</title>
<style>body{{margin:0;background:#F7F5F2;color:#14110E;
font-family:ui-monospace,Menlo,monospace;font-size:13px}}
.wrap{{max-width:620px;margin:0 auto;padding:24px}}</style>
</head><body><div class="wrap"><div id="interface-jsd"></div></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>window.PROCGREP_PALETTE={{paper:"#F7F5F2",ink:"#14110E",rule:"#d9d4cc",\
copper:"#CB4D20",blue:"#5692E5",teal:"#20A380",olive:"#585E53",gray:"#b7b1a7"}};</script>
<script>{d3charts}</script>
<script id="interface-jsd-data" type="application/json">{data}</script>
<script>{draw}</script>
</body></html>
"""

DRAW_JS = """
(function(){
  var el=document.getElementById('interface-jsd-data'); if(!el)return;
  var mount=document.getElementById('interface-jsd'); if(!mount||!window.ProcgrepCharts)return;
  var d=JSON.parse(el.textContent);
  ProcgrepCharts.matrix(mount,d.labels,d.values,{
    width:560,vmax:d.vmax,unit:'JSD',
    title:'Pairwise procedural divergence by interface',
    ariaLabel:'pairwise action-mix JSD between four coding interfaces'});
})();
"""

# Teal sequential ramp matching the canonical agent matrix (light tint -> deep teal).
TEAL = LinearSegmentedColormap.from_list("procgrep_teal", ["#dcebe5", "#3f9183", "#16544b"])


def main() -> int:
    mat = json.loads(RESULTS.read_text())["cross_interface_bpe"]["procedure_jsd_matrix"]
    labels = list(mat)
    z = [[mat[r][c] for c in labels] for r in labels]

    vmax = max(v for row in z for v in row)
    init()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(z, cmap=TEAL, vmin=0.0, vmax=vmax)

    # Print the JSD in each cell (parity with the plateau procedure_jsd matrix);
    # flip text to paper-white on dark cells for contrast, mute the zero diagonal.
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = z[i][j]
            color = OLIVE if i == j else ("#F7F5F2" if v > vmax * 0.55 else INK)
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.tick_params(colors=OLIVE, labelsize=9, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Pairwise procedural divergence by interface", color=INK, size=11, loc="left", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=OLIVE, labelsize=9, length=0)
    cbar.set_label("JSD", color=INK, size=10)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")

    # Interactive companion for the web essay, sharing the d3charts matrix.
    data = {"labels": labels, "values": z, "vmax": round(vmax, 4)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2) + "\n")
    d3charts = D3CHARTS.read_text()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(PAGE.format(d3charts=d3charts, data=json.dumps(data), draw=DRAW_JS))

    print(f"wrote {OUT}\n      {OUT_JSON}\n      {OUT_HTML}  (interfaces={labels})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
