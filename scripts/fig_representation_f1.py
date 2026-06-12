"""Intent: figure asset for the representation-comparison table in the essay, a
horizontal bar chart of retrieval F1 by representation, colored by source. Shows
the headline: structural/deterministic representations top the table, narrative
descriptions sit below, and the LLM divergence classifier ranges wildly. The
numbers are the table's; this only visualizes them. Emits a small JSON data file
plus a self-contained interactive D3 page that reuses the shared chart module.

Design decisions (benefit / price):
1. The values are passed in literally from the paper's table, not recomputed.
   Benefit: the figure matches the table exactly and stays the author's data.
   Price: if the table changes, update ROWS here too.
2. Point values draw as bars; the two ranges, the LLM classifier and the random
   baseline, draw as light bands with their span.
   Benefit: one frame shows both the ranking and the LLM's instability.
   Price: mixed mark types need a short legend, kept minimal.
3. Render through d3charts.barChart rather than a frozen SVG.
   Benefit: hover for exact values and a value-label toggle come for free, and
   the palette and chart style match the explorer and the other figures.
   Price: the figure ships as data plus module rather than a static SVG.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs" / "paper" / "data" / "representation_f1.json"
OUT_HTML = ROOT / "docs" / "figures" / "representation_f1.html"
D3CHARTS = ROOT / "docs" / "explorer" / "d3charts.js"

COPPER, OLIVE, GRAY, RULE = "#CB4D20", "#585E53", "#b7b1a7", "#d9d4cc"

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


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>retrieval F1 by representation</title>
<style>body{{margin:0;background:#F7F5F2;color:#14110E;
font-family:ui-monospace,Menlo,monospace;font-size:13px}}
.wrap{{max-width:700px;margin:0 auto;padding:24px}}</style>
</head><body><div class="wrap"><div id="representation-f1"></div></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script>window.PROCGREP_PALETTE={{paper:"#F7F5F2",ink:"#14110E",rule:"#d9d4cc",\
copper:"#CB4D20",blue:"#5692E5",teal:"#20A380",olive:"#585E53",gray:"#b7b1a7"}};</script>
<script>{d3charts}</script>
<script id="representation-f1-data" type="application/json">{data}</script>
<script>{draw}</script>
</body></html>
"""

# Shared draw routine, also emitted standalone so the essay can reuse it.
DRAW_JS = """
(function(){
  var el=document.getElementById('representation-f1-data'); if(!el)return;
  var mount=document.getElementById('representation-f1'); if(!mount||!window.ProcgrepCharts)return;
  var rows=JSON.parse(el.textContent).map(function(r){
    var base={label:r.label,color:r.color};
    if(r.lo!=null){
      base.lo=r.lo; base.hi=r.hi;
      base.valueLabel=r.lo.toFixed(2)+' to '+r.hi.toFixed(2);
      base.title=r.label+': F1 '+r.lo.toFixed(3)+' to '+r.hi.toFixed(3);
      base.tipHTML='<b>'+r.label+'</b><br>retrieval F1 <b>'+r.lo.toFixed(3)+
        '</b> to <b>'+r.hi.toFixed(3)+'</b>';
    }else{
      base.value=r.value;
      base.valueLabel=r.value.toFixed(3);
      base.title=r.label+': F1 '+r.value.toFixed(3);
      base.tipHTML='<b>'+r.label+'</b><br>retrieval F1 <b>'+r.value.toFixed(3)+'</b>';
    }
    return base;
  });
  ProcgrepCharts.barChart(mount,rows,{
    width:660,rowHeight:34,xMax:0.4,labelWidth:210,
    ariaLabel:'retrieval F1 by representation',
    xLabel:'retrieval F1 at k = 1, n = 289',
    xTicks:[{v:0,label:'0.0'},{v:0.1,label:'0.1'},{v:0.2,label:'0.2'},
      {v:0.3,label:'0.3'},{v:0.4,label:'0.4'}],
    legend:[{label:'deterministic',color:'#CB4D20'},
      {label:'varies by model',color:'#585E53'},
      {label:'LLM, unstable',color:'#b7b1a7'}]});
})();
"""


def main() -> int:
    data = []
    for label, val, grp in ROWS:
        row: dict[str, object] = {"label": label, "color": FILL[grp], "group": grp}
        if isinstance(val, tuple):
            row["lo"], row["hi"] = val
        else:
            row["value"] = val
        data.append(row)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2) + "\n")

    d3charts = D3CHARTS.read_text()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(PAGE.format(d3charts=d3charts, data=json.dumps(data), draw=DRAW_JS))
    print(f"wrote {OUT_JSON} and {OUT_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
