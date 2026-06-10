"""Build a self-contained static interface (one index.html, data embedded) over
the ecosystem catalog + per-dataset profiles. Opens with no server (file://).

    python case-studies/build_interface.py --catalog catalog.json \
        --profiles profile_nebius.json --out interface/index.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>procgrep · the agent-trace ecosystem</title>
<style>
:root{--paper:#F7F5F2;--ink:#14110E;--rule:#d9d4cc;--copper:#CB4D20;--blue:#5692E5;
--teal:#20A380;--olive:#585E53;--gray:#b7b1a7;--mono:ui-monospace,"SFMono-Regular",Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--mono);font-size:13px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:44px 32px 110px}
header{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--ink);padding-bottom:10px}
.mark{font-weight:600}.dim{color:var(--olive)}
nav span{color:var(--olive);margin-left:18px;cursor:pointer}
h1{font-family:Georgia,serif;font-weight:400;font-size:29px;line-height:1.25;max-width:24ch;margin:26px 0 6px}
.sub{color:var(--olive);max-width:62ch;margin:0 0 34px}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:11px;color:var(--olive);border-bottom:1px solid var(--rule);padding-bottom:6px;margin:34px 0 14px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule)}
.card{background:var(--paper);padding:16px 18px}.big{font-size:28px;font-weight:600}
.chip{display:inline-block;padding:1px 8px;margin:2px 5px 2px 0;border:1px solid var(--rule);border-radius:2px;font-size:11px;cursor:pointer}
.chip.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.chip.fmt{border-color:var(--copper);color:var(--copper)}.chip.fmt.on{background:var(--copper);color:#fff;border-color:var(--copper)}
.chip.un{background:#efeae2;border-color:var(--rule);color:var(--olive);cursor:default}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0 6px}
input,select{font-family:var(--mono);font-size:12px;padding:5px 8px;border:1px solid var(--rule);background:#fff;color:var(--ink)}
input{flex:1;min-width:180px}
label.gb{color:var(--olive);cursor:pointer;user-select:none}
table{width:100%;border-collapse:collapse;margin-top:6px}
th{text-align:left;font-weight:400;color:var(--olive);border-bottom:1px solid var(--ink);padding:6px 8px;cursor:pointer;user-select:none}
td{border-bottom:1px solid var(--rule);padding:7px 10px;vertical-align:top;font-variant-numeric:tabular-nums}
tr.clk:hover td{background:#efeae2}
.idcell{cursor:pointer}.idcell:hover{color:var(--copper)}
.barbg{display:inline-block;width:96px;height:9px;background:var(--rule);vertical-align:middle;position:relative}
.barbg .fill{position:absolute;left:0;top:0;height:9px}
.rrow{display:flex;align-items:center;gap:7px;margin:2px 0}
.rlab{width:74px;color:var(--olive);font-size:11px;text-align:right}
.rval{font-size:11px;width:34px}
.mix{display:inline-flex;height:10px;width:140px;border:1px solid var(--rule);vertical-align:middle}.mix>span{height:100%}
.detail{background:#efeae2;padding:10px 14px;font-size:12px}
.detail a{color:var(--copper)}
.grouphdr td{background:#efeae2;font-weight:600;border-bottom:1px solid var(--ink)}
.more{margin:14px 0;padding:7px 14px;border:1px solid var(--ink);background:none;cursor:pointer;font-family:var(--mono);font-size:12px}
.more:hover{background:var(--ink);color:var(--paper)}
.foot{margin-top:40px;border-top:1px solid var(--rule);padding-top:10px;color:var(--olive);font-size:11px}
.hidden{display:none}.spine{display:flex;flex-wrap:wrap;gap:3px;margin:10px 0}
.atom{font-size:10px;padding:1px 5px;border-radius:2px;color:#fff}
.qbox{width:100%;margin:4px 0 8px}
.qhit{padding:3px 2px;border-bottom:1px solid var(--rule);cursor:pointer}
.qhit:hover{color:var(--copper)}
.speed{color:var(--copper)}
.turn{border-left:2px solid var(--rule);padding:4px 10px;margin:6px 0}
.turn .role{color:var(--olive);text-transform:uppercase;font-size:10px;letter-spacing:.08em}
.back{color:var(--copper);cursor:pointer;margin-bottom:14px;display:inline-block}
</style></head><body><div class="wrap">
<header><span class="mark">procgrep</span><nav><span onclick="show('eco')">ecosystem</span><span onclick="show('why')">why structural</span><span class="dim">about</span></nav></header>

<div id="eco">
<h1>See how agents actually work — not whether they passed.</h1>
<p class="sub">A live map of public agent-trajectory datasets on Hugging Face. procgrep reads each
one's raw logs and rewrites them into a single vocabulary of actions (read a file, edit, run a
test, search…) so datasets in different formats become directly comparable. Click any dataset to
look inside.</p>

<div class="eyebrow">findings</div>
<div class="cards" id="findings"></div>

<div class="eyebrow">the index</div>
<div class="controls">
  <input id="q" placeholder="filter datasets…" oninput="ST.q=this.value;ST.page=1;render()">
  <label class="gb"><input type="checkbox" onchange="ST.group=this.checked;render()" style="flex:none"> group by author</label>
  <span id="filters" class="dim"></span>
</div>
<table id="index"><thead><tr>
<th data-k="id">dataset</th><th>format</th><th data-k="downloads">downloads</th>
<th data-k="likes">likes</th><th data-k="last_modified">updated</th></tr></thead><tbody></tbody></table>
<button class="more hidden" id="more" onclick="ST.page++;render()">view more</button>
<div class="foot" id="foot"></div>
</div>

<div id="ds" class="hidden"></div>
<div id="tr" class="hidden"></div>
<div id="why" class="hidden"></div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const CAT=D.catalog, DS=CAT.datasets, PROF=D.profiles;
const ATOM={search_repo:'#585E53',read_file:'#5692E5',edit:'#CB4D20',create_file:'#3D7AD8',
run_test:'#20A380',submit:'#14110E',think:'#b7b1a7',localize:'#8C1040',delete_file:'#A03D18',error:'#B4184F',other:'#d9d4cc'};
const pct=x=>x==null?'—':(x*100).toFixed(0)+'%';
const fmt=n=>n>=1000?(n/1000).toFixed(n>=10000?0:1)+'k':String(n);
const ST={q:'',sort:'downloads',dir:-1,group:false,page:1,size:30,fmtFilter:null,statusFilter:null,open:new Set()};

function statusOf(r){return r.supported?'supported':(r.out_of_scope?'out-of-scope':'unsupported');}
function dupOf(r){const p=PROF[r.id];return p?p.redundancy.exact_dup_rate:null;}
function gapOf(r){const p=PROF[r.id];return p?p.redundancy.coverage_diverse-p.redundancy.coverage_shortest:null;}

function findings(){
  const sniffed=DS.length, sup=DS.filter(r=>r.supported).length;
  const byA={}; DS.forEach(r=>{if(r.supported)byA[r.adapter]=(byA[r.adapter]||0)+1;});
  const fmtChips=Object.entries(byA).sort((a,b)=>b[1]-a[1]).map(([a,n])=>
    `<span class="chip fmt ${ST.fmtFilter===a?'on':''}" onclick="ST.fmtFilter=ST.fmtFilter===\'${a}\'?null:\'${a}\';ST.page=1;render()">${a} ${n}</span>`).join('');
  const red=Object.values(PROF).map(p=>{const s=p.redundancy.coverage_shortest,d=p.redundancy.coverage_diverse;
    return `<div style="margin:9px 0 3px"><span class="dim">${p.dataset.split('/').pop()}</span></div>
      <div class="rrow"><span class="rlab">keep shortest</span><span class="barbg"><span class="fill" style="width:${s*100}%;background:var(--olive)"></span></span><span class="rval">${pct(s)}</span></div>
      <div class="rrow"><span class="rlab">keep diverse</span><span class="barbg"><span class="fill" style="width:${d*100}%;background:var(--copper)"></span></span><span class="rval">${pct(d)}</span></div>`;}).join('');
  document.getElementById('findings').innerHTML=
   `<div class="card"><div class="dim">coverage</div><div class="big">${sup}/${sniffed}</div>
      <div class="dim">parseable, of ${CAT.n_discovered} discovered</div></div>
    <div class="card"><div class="dim">formats (click to filter)</div><div style="margin-top:8px">${fmtChips}</div></div>
    <div class="card"><div class="dim">trim a dataset to a fixed budget — how much of its procedural variety survives if you keep the shortest traces (common practice) vs the most diverse (procgrep)</div>${red||'<span class="dim">profiles pending</span>'}</div>`;
}

function rowsFiltered(){
  let rows=DS.filter(r=>r.id.toLowerCase().includes(ST.q.toLowerCase()));
  if(ST.fmtFilter)rows=rows.filter(r=>r.adapter===ST.fmtFilter);
  if(ST.statusFilter)rows=rows.filter(r=>statusOf(r)===ST.statusFilter);
  const key=ST.sort;
  rows.sort((a,b)=>{let va,vb;
    if(key==='dup'){va=dupOf(a)??-1;vb=dupOf(b)??-1;}
    else if(key==='gap'){va=gapOf(a)??-1;vb=gapOf(b)??-1;}
    else{va=a[key]??'';vb=b[key]??'';}
    return (va<vb?-1:va>vb?1:0)*ST.dir;});
  return rows;
}
function tagChips(r){
  return r.adapter?`<span class="chip fmt">${r.adapter}</span>`
    :`<span class="chip un">${r.out_of_scope?'out of scope':'unsupported'}</span>`;
}
function rowHTML(r){
  const p=PROF[r.id];
  let h=`<tr class="clk"><td class="idcell" onclick="toggle('${r.id}')">${r.id}</td><td>${tagChips(r)}</td>
    <td>${fmt(r.downloads)}</td><td>${r.likes||0}</td><td class="dim">${r.last_modified||'—'}</td></tr>`;
  if(ST.open.has(r.id)){
    const why=r.adapter?`format <b>${r.adapter}</b> (confidence ${r.confidence})`:(r.error?`unsupported — <span class="dim">${r.error}</span>`:'out of scope (no discrete tool actions)');
    const red=p?` · redundancy: ${pct(p.redundancy.exact_dup_rate)} exact-dup, shortest keeps ${pct(p.redundancy.coverage_shortest)} vs diverse ${pct(p.redundancy.coverage_diverse)}`:'';
    const prof=p?` · <a onclick="dataset('${r.id}')" style="cursor:pointer">open profile →</a>`:'';
    h+=`<tr><td colspan="5" class="detail"><a href="https://huggingface.co/datasets/${r.id}" target="_blank">huggingface.co/datasets/${r.id} ↗</a>
       · ${why}${red}${prof}</td></tr>`;}
  return h;
}
function render(){
  findings();
  const rows=rowsFiltered();
  const tb=document.querySelector('#index tbody');
  if(ST.group){
    const g={}; rows.forEach(r=>(g[r.author]=g[r.author]||[]).push(r));
    tb.innerHTML=Object.entries(g).sort((a,b)=>b[1].length-a[1].length).map(([au,rs])=>
      `<tr class="grouphdr"><td colspan="5">${au} · ${rs.length}</td></tr>`+rs.map(rowHTML).join('')).join('');
    document.getElementById('more').classList.add('hidden');
  }else{
    const shown=rows.slice(0,ST.page*ST.size);
    tb.innerHTML=shown.map(rowHTML).join('');
    document.getElementById('more').classList.toggle('hidden',shown.length>=rows.length);
  }
  document.getElementById('filters').innerHTML=
    ['supported','unsupported','out-of-scope'].map(s=>`<span class="chip ${ST.statusFilter===s?'on':''}" onclick="ST.statusFilter=ST.statusFilter===\'${s}\'?null:\'${s}\';ST.page=1;render()">${s}</span>`).join('');
  document.getElementById('foot').textContent=
    `catalog generated ${CAT.generated||'—'} · ${CAT.n_discovered} datasets discovered · ${DS.length} sniffed · showing ${rows.length} after filters`;
}
function toggle(id){ST.open.has(id)?ST.open.delete(id):ST.open.add(id);render();}
document.querySelectorAll('#index th[data-k]').forEach(th=>th.onclick=()=>{
  const k=th.dataset.k;ST.dir=(k===ST.sort)?-ST.dir:-1;ST.sort=k;render();});

function mixbar(m){return '<span class="mix">'+Object.entries(m).map(([a,v])=>
  `<span style="width:${v*140}px;background:${ATOM[a]||'#ccc'}" title="${a} ${pct(v)}"></span>`).join('')+'</span>';}

// behaviour-search sample queries: structural predicates over the atom spine.
const QUERIES=[
 {id:'streak',label:'edit-streak ≥5',f:a=>/(?:edit ){5,}/.test(a.join(' ')+' ')},
 {id:'nosub',label:'submitted without testing',f:a=>a.includes('submit')&&!a.includes('run_test')},
 {id:'td',label:'test-driven (test before first edit)',f:a=>{const r=a.indexOf('run_test'),e=a.indexOf('edit');return r>=0&&(e<0||r<e);}},
 {id:'stuck',label:'stuck reading',f:a=>/(?:read_file (?:think )?){4,}/.test(a.join(' ')+' ')},
 {id:'loop',label:'canonical resolve loop',f:a=>/search_repo read_file edit run_test/.test(a.join(' '))},
 {id:'nosearch',label:'never searched the repo',f:a=>!a.includes('search_repo')},
 {id:'repro',label:'wrote a repro script',f:a=>{const c=a.indexOf('create_file');return c>=0&&a.lastIndexOf('submit')>c;}},
 {id:'recover',label:'recovered from an error',f:a=>/error edit/.test(a.join(' '))},
];
let CURDS=null;
function runQ(custom,id){
  const pool=PROF[CURDS].atoms_pool||[]; let f,label;
  if(id){const q=QUERIES.find(x=>x.id===id);f=q.f;label=q.label;document.getElementById('qbox').value='';}
  else if(custom){let rx;try{rx=new RegExp(custom);}catch(e){document.getElementById('qres').innerHTML='<span class="dim">invalid pattern</span>';return;}
    f=a=>rx.test(a.join(' ')+' ');label='/'+custom+'/';}
  else{document.getElementById('qres').innerHTML='';return;}
  const t=performance.now();const hits=pool.filter(x=>f(x.atoms));const ms=performance.now()-t;
  document.getElementById('qres').innerHTML=
   `<b>${hits.length}</b> / ${pool.length} traces match <b>${label}</b> · `+
   `<span class="speed">scanned in ${ms.toFixed(1)} ms — no model call</span><br>`+
   hits.slice(0,25).map(h=>`<div class="qhit" onclick="poolTrace('${h.trace_id}')">${h.model.split('-').pop()} · ${h.atoms.length} steps · <span class="dim">${h.atoms.slice(0,20).join(' ')}${h.atoms.length>20?' …':''}</span></div>`).join('');
}
function poolTrace(tid){const p=PROF[CURDS];const s=p.samples.find(x=>x.trace_id===tid);
  if(s){const i=p.samples.indexOf(s);trace(CURDS,i);return;}
  const h=(p.atoms_pool||[]).find(x=>x.trace_id===tid);if(!h)return;show('tr');
  const spine=h.atoms.map(a=>`<span class="atom" style="background:${ATOM[a]||'#ccc'}">${a}</span>`).join('');
  document.getElementById('tr').innerHTML=`<span class="back" onclick="dataset('${CURDS}')">← ${CURDS.split('/').pop()}</span>
    <h1 style="font-size:19px">${h.model} · ${h.atoms.length} steps</h1>
    <div class="eyebrow">procedural spine</div><div class="spine">${spine}</div>
    <div class="dim">conversation not sampled for this trace</div>`;}

function dataset(id){const p=PROF[id];CURDS=id;show('ds');
  const models=Object.entries(p.by_model).map(([m,s])=>
    `<tr><td>${m}</td><td>${s.n}</td><td>${s.median_len}</td><td>${pct(s.exact_dup_rate)}</td><td>${mixbar(s.action_mix)}</td></tr>`).join('');
  const samples=p.samples.map((s,i)=>`<span class="chip" onclick="trace('${id}',${i})">trace ${i+1} · ${s.model.split('-').pop()} · ${s.atoms.length} steps</span>`).join('');
  document.getElementById('ds').innerHTML=
   `<span class="back" onclick="show('eco')">← ecosystem</span><h1 style="font-size:21px">${id}</h1>
    <p class="sub">${p.adapter} · ${p.n_traces} traces · ${p.n_models} models · exact-dup ${pct(p.redundancy.exact_dup_rate)} · shortest keeps ${pct(p.redundancy.coverage_shortest)} vs diverse ${pct(p.redundancy.coverage_diverse)}</p>
    <div class="eyebrow">by model</div><table><thead><tr><th>model</th><th>n</th><th>median len</th><th>exact-dup</th><th>action mix</th></tr></thead><tbody>${models}</tbody></table>
    <div class="eyebrow">search by behaviour <span class="dim" style="text-transform:none;letter-spacing:0">— structural predicates over the action spine; instant, no model call</span></div>
    <input class="qbox" id="qbox" placeholder="atom pattern, e.g.  (edit ){5,}" oninput="runQ(this.value)">
    <div>${QUERIES.map(q=>`<span class="chip" onclick="runQ(null,'${q.id}')">${q.label}</span>`).join('')}</div>
    <div id="qres" style="margin-top:10px"></div>
    <div class="eyebrow">sampled traces (click to open the conversation)</div><div>${samples}</div>`;}
function trace(id,i){const s=PROF[id].samples[i];show('tr');
  const spine=s.atoms.map(a=>`<span class="atom" style="background:${ATOM[a]||'#ccc'}">${a}</span>`).join('');
  const turns=s.turns.map(t=>{const body=t.tools?('⚙ '+t.tools.join(', ')):(t.text||'');
    return `<div class="turn"><span class="role">${t.role}</span><br>${(body+'').replace(/</g,'&lt;')}</div>`;}).join('');
  document.getElementById('tr').innerHTML=
   `<span class="back" onclick="dataset('${id}')">← ${id.split('/').pop()}</span><h1 style="font-size:19px">${s.model} · ${s.atoms.length} steps</h1>
    <div class="eyebrow">procedural spine</div><div class="spine">${spine}</div><div class="eyebrow">conversation</div>${turns}`;}
function whyView(){const e=D.experiment;const el=document.getElementById('why');
  if(!e){el.innerHTML='<span class="back" onclick="show(\'eco\')">← ecosystem</span><p class="sub">experiment not loaded.</p>';return;}
  const ratio=Math.round(e.totals.mean_llm_latency_s*1e6/e.procgrep_us_per_decision);
  const struct=Object.entries(e.predicates).filter(([k,v])=>v.kind==='structural');
  const models=Object.keys(struct[0][1].judges);
  const head=`<th>predicate</th>${models.map(m=>`<th>${m.split('/').pop()}</th>`).join('')}<th>inter-judge κ</th>`;
  const body=struct.map(([k,v])=>`<tr><td>${k}</td>${models.map(m=>{const a=v.judges[m].accuracy;
    return `<td>${a==null?'—':a.toFixed(2)}</td>`;}).join('')}<td>${v.kappa==null?'—':v.kappa}</td></tr>`).join('');
  const fz=Object.entries(e.predicates).find(([k,v])=>v.kind==='fuzzy');
  el.innerHTML=
   `<span class="back" onclick="show('eco')">← ecosystem</span>
    <h1>Why not just ask an LLM?</h1>
    <p class="sub">Every behavioural question here — "did it edit 5× in a row?", "did it submit without testing?" — can be asked two ways: a deterministic structural query over the action spine, or an LLM judge over the trace. We ran both over the same trajectories.</p>
    <div class="cards">
      <div class="card"><div class="dim">speed</div><div class="big">${ratio.toLocaleString()}×</div>
        <div class="dim">procgrep ${e.procgrep_us_per_decision} µs vs LLM ${e.totals.mean_llm_latency_s}s per decision · $0 vs API cost</div></div>
      <div class="card"><div class="dim">accuracy (chance = .50)</div><div class="big">≈ chance</div>
        <div class="dim">LLM judges are at chance on counting/order predicates — they can't track structure over a long trace</div></div>
      <div class="card"><div class="dim">reliability</div><div class="big">κ ≈ 0</div>
        <div class="dim">judges barely agree with each other — on structural facts and fuzzy ones alike</div></div>
    </div>
    <div class="eyebrow">LLM judge accuracy vs the exact structural answer · balanced samples, chance = 0.50</div>
    <table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
    <p class="note">Fuzzy property (no deterministic ground truth) — <b>${fz?fz[0]:''}</b>: inter-judge κ = ${fz?fz[1].kappa:'—'}. procgrep does ${e.procgrep_ms_full} ms for ${e.corpus} traces × ${struct.length} predicates; the LLM route is ${ratio.toLocaleString()}× slower, costs per call, and is no more reliable. ${e.totals.judge_calls} judge calls, ${e.totals.total_tokens.toLocaleString()} tokens.</p>`;}
function show(id){['eco','ds','tr','why'].forEach(x=>document.getElementById(x).classList.toggle('hidden',x!==id));
  if(id==='why')whyView();if(id==='eco')window.scrollTo(0,0);}
render();
if(location.hash==='#why')show('why');
else if(location.hash.startsWith('#ds=')){const [id,q]=location.hash.slice(4).split('&q=');
  dataset(decodeURIComponent(id));if(q)runQ(null,q);}
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--profiles", nargs="*", default=[])
    ap.add_argument("--experiment", default=None, help="query_vs_llm_full.json for the 'why structural' page")
    ap.add_argument("--out", default="interface/index.html")
    args = ap.parse_args()

    catalog = json.loads(Path(args.catalog).read_text())
    if isinstance(catalog, list):  # tolerate the old flat-list catalog
        catalog = {"generated": "", "n_discovered": len(catalog), "datasets": catalog}
    profiles = {}
    for path in args.profiles:
        p = json.loads(Path(path).read_text())
        profiles[p["dataset"]] = p
    experiment = json.loads(Path(args.experiment).read_text()) if args.experiment else None

    data = json.dumps({"catalog": catalog, "profiles": profiles, "experiment": experiment})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", data))
    print(f"wrote {out}  ({len(catalog['datasets'])} datasets, {len(profiles)} profiles)")


if __name__ == "__main__":
    main()
