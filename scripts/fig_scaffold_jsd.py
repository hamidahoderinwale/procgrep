"""Intent: generate the scaffold-vs-model figure for the essay, a log-axis dot
plot of action-mix Jensen-Shannon divergence between trace groups. It shows the
headline decomposition: within one scaffold the model barely moves the
fingerprint, while crossing scaffolds moves it one to two orders of magnitude
more. Reads the local spine store, emits the row data as JSON, and renders an
interactive D3 version using the shared chart module.

Design decisions (benefit / price):
1. Compute JSD from the store at build time, do not hardcode the numbers.
   Benefit: the figure regenerates and stays honest if the store changes.
   Price: needs the local parquet present to rebuild. When the store is absent
   we fall back to the last computed values committed alongside this script, so
   the interactive figure still builds in a checkout without the data.
2. Emit a small JSON data file plus a self-contained interactive HTML that
   loads the shared d3charts.js module.
   Benefit: one chart implementation shared with the explorer and essay; hover
   for exact values and a value-label toggle come for free; matches palette.
   Price: the figure now ships as data + module rather than a frozen SVG.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "procgrep_spines.parquet"
OUT_JSON = ROOT / "docs" / "paper" / "data" / "scaffold_jsd.json"
OUT_HTML = ROOT / "docs" / "figures" / "scaffold_jsd.html"
D3CHARTS = ROOT / "docs" / "explorer" / "d3charts.js"

INK, OLIVE, RULE, COPPER, TEAL = "#14110E", "#585E53", "#d9d4cc", "#CB4D20", "#20A380"

# Last computed values, used only when the local parquet store is absent so the
# interactive figure still builds in a checkout without the data.
FALLBACK = [
    ("Llama 8B vs 70B, one scaffold", 0.0013, "within"),
    ("Llama 8B vs 405B, one scaffold", 0.0043, "within"),
    ("OpenHands vs OpenHands variant", 0.1304, "across"),
    ("SWE-agent vs OpenHands", 0.1644, "across"),
    ("SWE-agent vs OpenHands variant", 0.3281, "across"),
]


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


def _rows_from_store() -> list[tuple[str, float, str]]:
    import pandas as pd

    df = pd.read_parquet(STORE)

    def mix_where(dataset: str, agent: str | None = None) -> dict[str, float]:
        sub = df[df.dataset == dataset]
        if agent:
            sub = sub[sub.agent == agent]
        return _mix(list(sub.spine))

    ne = "nebius/SWE-agent-trajectories"
    return [
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


# Self-contained interactive page: D3 from the CDN, the shared module inlined,
# and the row data rendered as a log-x dot plot with two color groups.
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scaffold vs model divergence</title>
<style>body{{margin:0;background:#F7F5F2;color:#14110E;
font-family:ui-monospace,Menlo,monospace;font-size:13px}}
.wrap{{max-width:680px;margin:0 auto;padding:24px}}</style>
</head><body><div class="wrap"><div id="scaffold-jsd"></div></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>window.PROCGREP_PALETTE={{paper:"#F7F5F2",ink:"#14110E",rule:"#d9d4cc",\
copper:"#CB4D20",blue:"#5692E5",teal:"#20A380",olive:"#585E53",gray:"#b7b1a7"}};</script>
<script>{d3charts}</script>
<script id="scaffold-jsd-data" type="application/json">{data}</script>
<script>{draw}</script>
</body></html>
"""

# Shared draw routine, also emitted standalone so the essay can reuse it.
DRAW_JS = """
(function(){
  var el=document.getElementById('scaffold-jsd-data'); if(!el)return;
  var mount=document.getElementById('scaffold-jsd'); if(!mount||!window.ProcgrepCharts)return;
  var rows=JSON.parse(el.textContent).map(function(r){
    var color=r.group==='within'?'#20A380':'#CB4D20';
    return {label:r.label,x:Math.max(r.jsd,8e-4),color:color,valueLabel:r.jsd.toFixed(3),
      title:r.label+': JSD '+r.jsd.toFixed(4),
      tipHTML:'<b>'+r.label+'</b><br>action-mix JSD <b>'+r.jsd.toFixed(4)+'</b>'+
        '<br><span style="color:'+(r.group==='within'?'#7fd0bc':'#e89b82')+'">'+r.group+' scaffold</span>'};
  });
  ProcgrepCharts.logDotPlot(mount,rows,{
    width:640,rowHeight:38,xDomain:[8e-4,0.5],
    ariaLabel:'scaffold versus model divergence, log x',
    xLabel:'action-mix JSD, log scale',
    xTicks:[{v:1e-3,label:'0.001'},{v:1e-2,label:'0.01'},{v:1e-1,label:'0.1'}]});
})();
"""


def main() -> int:
    if STORE.exists():
        rows = _rows_from_store()
        source = "store"
    else:
        rows = FALLBACK
        source = "fallback"

    data = [{"label": lab, "jsd": round(v, 4), "group": grp} for lab, v, grp in rows]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2) + "\n")

    d3charts = D3CHARTS.read_text()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(PAGE.format(d3charts=d3charts, data=json.dumps(data), draw=DRAW_JS))
    print(f"wrote {OUT_JSON} and {OUT_HTML}  (source={source}, {[d['jsd'] for d in data]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
