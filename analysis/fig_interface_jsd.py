"""Intent: render the cross-interface procedural-divergence matrix for the essay,
the interface counterpart to fig_jsd_matrix_full_canonical (which grades agents).
Adds interface as a fourth fingerprint axis alongside lineage, family, and
scaffold: pairwise action-mix JSD between four coding interfaces.

Design decisions (benefit / price):
1. Read the matrix from a committed data file (docs/paper/data/interface_jsd_matrix.json)
   rather than recompute. Benefit: self-contained and reproducible from the repo
   alone. Price: the divergence numbers are produced upstream of this figure and
   checked in as the validated asset; this figure only renders them.
2. Match the canonical agent matrix exactly: Instrument Sans labels and the shared saved-data renderer.
   Preserve the interactive companion outputs for existing callers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from render_essay_figures import render_one

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS = ROOT / "docs" / "paper" / "data" / "interface_jsd_matrix.json"
OUT = ROOT / "docs/figures/fig_interface_jsd_matrix/fig_interface_jsd_matrix.png"
OUT_JSON = ROOT / "docs" / "paper" / "data" / "interface_jsd.json"
OUT_HTML = ROOT / "docs" / "figures" / "interface_jsd.html"
D3CHARTS = ROOT / "docs" / "explorer" / "d3charts.js"

# Self-contained interactive page mirroring fig_scaffold_jsd: D3 from the CDN,
# the shared module inlined, the matrix rendered with hover-for-exact-value.
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cross-interface procedural divergence</title>
<link rel="stylesheet" href="../assets/fonts.css">
<style>body{{margin:0;background:#F7F5F2;color:#14110E;
font-family:"Instrument Sans",Arial,sans-serif;font-size:13px}}
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


def main() -> int:
    mat = json.loads(RESULTS.read_text())
    labels = list(mat)
    z = [[mat[r][c] for c in labels] for r in labels]

    vmax = max(v for row in z for v in row)
    render_one("fig_interface_jsd_matrix")

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
