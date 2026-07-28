"""Intent: interactive version of distillation Panel A, the per-trajectory action
entropy distributions of the Claude-3.7 parent versus the SWE-agent-LM-32B child
it was distilled into, at both the canonical and native atom layers. It shows the
headline: the child's distribution concentrates and drops below the parent's, the
structural signature of distillation narrowing onto a few action loops. Reuses the
exact per-trajectory Shannon-entropy computation from the static figure, so the
interactive figure reproduces the PNG rather than restating it.

Reads the fingerprint files from the sibling procgrep-audits repo (the same path
the distillation case study falls back to) and writes the box summary stats to
docs/paper/data/distillation_entropy.json plus a self-contained interactive D3
page. The essay reads the committed JSON, so the build needs no raw data present.

Design decisions (benefit / price):
1. Compute per-trajectory entropy with the same Counter-over-atoms method as
   the static figure, then emit only box summary stats.
   Benefit: the interactive figure matches the published PNG exactly, and the
   committed JSON carries no raw trajectories.
   Price: if the raw fingerprints move, update FP_DIRS here.
2. Render through d3charts.boxPlot, mapping parent to the editorial blue and
   child to teal, preserving the static figure's parent-blue / child-green
   reading inside the essay palette.
   Benefit: hover for the full five-number summary and a mean-label toggle come
   for free, and the chart matches the other interactive figures.
   Price: the figure ships as data plus module rather than a frozen PNG.
"""

from __future__ import annotations

import json
import math
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs" / "paper" / "data" / "distillation_entropy.json"
OUT_HTML = ROOT / "docs" / "figures" / "distillation_entropy.html"
D3CHARTS = ROOT / "docs" / "explorer" / "d3charts.js"

# Where the parent/child fingerprint files live, in preference order. The sibling
# procgrep-audits repo is the canonical home, matching the case study's fallback.
FP_DIRS = [
    ROOT / "results",
    ROOT.parent / "procgrep-audits" / "results",
]
PARENT_FILE = "fingerprints_claude37_parent_n300.jsonl"
CHILD_FILE = "fingerprints_child_n500.jsonl"

BLUE, TEAL = "#5692E5", "#20A380"  # parent, child


def _resolve(name: str) -> Path:
    for d in FP_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found under {[str(d) for d in FP_DIRS]}")


def _per_traj_entropy(path: Path, layer: str) -> list[float]:
    out: list[float] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        atoms = json.loads(line).get(layer, [])
        if isinstance(atoms, str):
            atoms = json.loads(atoms.replace("'", '"'))
        if not atoms:
            continue
        n = len(atoms)
        out.append(-sum((v / n) * math.log2(v / n) for v in Counter(atoms).values()))
    return out


def _quantile(xs: list[float], p: float) -> float:
    """Linear-interpolation quantile, matching pandas' default."""
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    idx = p * (len(s) - 1)
    lo = math.floor(idx)
    frac = idx - lo
    if lo + 1 >= len(s):
        return s[lo]
    return s[lo] + frac * (s[lo + 1] - s[lo])


def _box(values: list[float], panel: str, group: str, color: str) -> dict:
    mean = st.mean(values)
    return {
        "panel": panel,
        "group": group,
        "color": color,
        "n": len(values),
        "mean": round(mean, 3),
        "med": round(_quantile(values, 0.5), 3),
        "q1": round(_quantile(values, 0.25), 3),
        "q3": round(_quantile(values, 0.75), 3),
        "lo": round(_quantile(values, 0.05), 3),
        "hi": round(_quantile(values, 0.95), 3),
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>distillation: action entropy, parent vs child</title>
<style>body{{margin:0;background:#F7F5F2;color:#14110E;
font-family:ui-monospace,Menlo,monospace;font-size:13px}}
.wrap{{max-width:560px;margin:0 auto;padding:24px}}</style>
</head><body><div class="wrap"><div id="distillation-entropy"></div></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>window.PROCGREP_PALETTE={{paper:"#F7F5F2",ink:"#14110E",rule:"#d9d4cc",\
copper:"#CB4D20",blue:"#5692E5",teal:"#20A380",olive:"#585E53",gray:"#b7b1a7"}};</script>
<script>{d3charts}</script>
<script id="distillation-entropy-data" type="application/json">{data}</script>
<script>{draw}</script>
</body></html>
"""

# Shared draw routine, also emitted into the essay so both render identically.
DRAW_JS = """
(function(){
  var el=document.getElementById('distillation-entropy-data'); if(!el)return;
  var mount=document.getElementById('distillation-entropy'); if(!mount||!window.ProcgrepCharts)return;
  var boxes=JSON.parse(el.textContent).map(function(b){
    b.tipHTML='<b>'+b.group+', '+b.panel+'</b><br>mean <b>'+b.mean.toFixed(2)+'</b> bits'+
      '<br>median '+b.med.toFixed(2)+', IQR '+b.q1.toFixed(2)+' to '+b.q3.toFixed(2)+
      '<br>5 to 95 pct '+b.lo.toFixed(2)+' to '+b.hi.toFixed(2)+'<br>n = '+b.n;
    return b;
  });
  ProcgrepCharts.boxPlot(mount,boxes,{
    width:520,height:280,panels:['canonical','native'],groups:['Parent','Child'],
    yDomain:[1.0,2.8],yTicks:[1.0,1.4,1.8,2.2,2.6],
    yLabel:'per-trajectory entropy, bits',
    ariaLabel:'action entropy distributions, parent versus child, canonical and native'});
})();
"""


def main() -> int:
    parent = _resolve(PARENT_FILE)
    child = _resolve(CHILD_FILE)
    boxes = []
    for panel, layer in (("canonical", "atoms_canonical"), ("native", "atoms_native")):
        boxes.append(_box(_per_traj_entropy(parent, layer), panel, "Parent", BLUE))
        boxes.append(_box(_per_traj_entropy(child, layer), panel, "Child", TEAL))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(boxes, indent=2) + "\n")

    d3charts = D3CHARTS.read_text()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(PAGE.format(d3charts=d3charts, data=json.dumps(boxes), draw=DRAW_JS))
    diff = boxes[0]["mean"] - boxes[1]["mean"]
    print(f"wrote {OUT_JSON} and {OUT_HTML}  (canonical mean parent-child = {diff:.3f} bits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
