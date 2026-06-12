"""Build a self-contained static interface (one index.html, data embedded) over
the ecosystem catalog + per-dataset profiles. Opens with no server (file://).

Query-first: the structural-query playground is the landing/hero; the ecosystem
catalog is demoted to a dataset picker + a "browse all" view. Four tabs:
Query · Curate · Per-model · Why structural. Zero JS dependencies.

    python case-studies/build_interface.py --catalog catalog.json \
        --profiles profile_nebius.json --experiment query_vs_llm_full.json \
        --logos logos.json --out interface/index.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>procgrep · grep for agent traces</title>
<style>
:root{--paper:#F7F5F2;--ink:#14110E;--rule:#d9d4cc;--copper:#CB4D20;--blue:#5692E5;
--teal:#20A380;--olive:#585E53;--gray:#b7b1a7;--mono:ui-monospace,"SFMono-Regular",Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--mono);font-size:13px;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:44px 32px 110px}
header{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid var(--ink);padding-bottom:10px}
.mark{font-weight:600}.dim{color:var(--olive)}
nav span{color:var(--olive);margin-left:18px;cursor:pointer}
nav span.act{color:var(--ink);border-bottom:2px solid var(--copper);padding-bottom:8px}
h1{font-family:Georgia,serif;font-weight:400;font-size:29px;line-height:1.25;max-width:26ch;margin:26px 0 6px}
.sub{color:var(--olive);max-width:64ch;margin:0 0 26px}
.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:11px;color:var(--olive);border-bottom:1px solid var(--rule);padding-bottom:6px;margin:30px 0 14px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--rule);border:1px solid var(--rule)}
.card{background:var(--paper);padding:16px 18px}.big{font-size:28px;font-weight:600}
.chip{display:inline-block;padding:2px 9px;margin:3px 6px 3px 0;border:1px solid var(--rule);border-radius:2px;font-size:11px;cursor:pointer}
.chip:hover{border-color:var(--ink)}
.chip.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.chip.fmt{border-color:var(--copper);color:var(--copper)}.chip.fmt.on{background:var(--copper);color:#fff}
.chip.un{background:#efeae2;border-color:var(--rule);color:var(--olive);cursor:default}
.chip.gap{border-color:var(--copper);color:var(--copper);background:#fff;cursor:default}
.bstrip{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 2px}
.bcard{display:inline-flex;align-items:center;gap:7px;padding:6px 11px;border:1px solid var(--rule);border-radius:3px;cursor:pointer;background:#fff}
.bcard.on{border-color:var(--ink);background:#efeae2}
.blogo{width:16px;height:16px;object-fit:contain;vertical-align:middle;border-radius:2px}
.bmono{display:inline-flex;width:16px;height:16px;align-items:center;justify-content:center;background:var(--olive);color:#fff;font-size:9px;border-radius:2px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0 6px}
input,select{font-family:var(--mono);font-size:12px;padding:5px 8px;border:1px solid var(--rule);background:#fff;color:var(--ink)}
label.gb{color:var(--olive);cursor:pointer;user-select:none}
table{width:100%;border-collapse:collapse;margin-top:6px}
th{text-align:left;font-weight:400;color:var(--olive);border-bottom:1px solid var(--ink);padding:6px 8px;cursor:pointer;user-select:none}
td{border-bottom:1px solid var(--rule);padding:7px 10px;vertical-align:top;font-variant-numeric:tabular-nums}
tr.clk:hover td{background:#efeae2}
.idcell{cursor:pointer}.idcell:hover{color:var(--copper)}
.barbg{display:inline-block;width:120px;height:9px;background:var(--rule);vertical-align:middle;position:relative}
.barbg .fill{position:absolute;left:0;top:0;height:9px}
.rrow{display:flex;align-items:center;gap:8px;margin:3px 0}
.rlab{width:120px;color:var(--olive);font-size:11px;text-align:right}
.rval{font-size:11px;width:40px}
.mix{display:inline-flex;height:11px;width:160px;border:1px solid var(--rule);vertical-align:middle}.mix>span{height:100%}
.detail{background:#efeae2;padding:10px 14px;font-size:12px}
.detail a{color:var(--copper)}
.grouphdr td{background:#efeae2;font-weight:600;border-bottom:1px solid var(--ink);cursor:pointer}
.more{margin:14px 0;padding:7px 14px;border:1px solid var(--ink);background:none;cursor:pointer;font-family:var(--mono);font-size:12px}
.more:hover{background:var(--ink);color:var(--paper)}
.foot{margin-top:40px;border-top:1px solid var(--rule);padding-top:10px;color:var(--olive);font-size:11px}
.hidden{display:none}.spine{display:flex;flex-wrap:wrap;gap:3px;margin:10px 0}
.atom{font-size:10px;padding:1px 5px;border-radius:2px;color:#fff}
.atom.nz{opacity:.4}
body.hide-noise .atom.nz{display:none}
.gap{font-family:var(--mono);font-size:10px;color:var(--gray);padding:1px 3px;align-self:center}
body.hide-noise .gap{display:none}
.qbar{display:flex;gap:0;margin:6px 0 10px;border:2px solid var(--ink);background:#fff}
.qbar span.mag{padding:9px 4px 9px 12px;color:var(--olive)}
#q{flex:1;border:none;font-size:16px;padding:10px 12px 10px 4px;background:transparent}
#q:focus{outline:none}
.tryline{margin:2px 0 16px;color:var(--olive)}
.dsbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:2px 0 18px;color:var(--olive)}
.qhit{padding:5px 2px;border-bottom:1px solid var(--rule);cursor:pointer;display:flex;align-items:center;gap:8px}
.qhit:hover{color:var(--copper)}
.qhmodel{min-width:96px}
.spinemini{display:inline-flex;gap:1px;flex-wrap:wrap;vertical-align:middle;flex:1}
.spinemini i{display:inline-block;width:5px;height:12px;border-radius:1px}
.spinemini i.nz{opacity:.26}
body.hide-noise .spinemini i.nz{display:none}
.noisetog{border-color:var(--olive);color:var(--olive)}
body.hide-noise .noisetog{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.speed{color:var(--copper)}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0;color:var(--olive);font-size:11px}
.legend span i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;vertical-align:baseline}
.turn{border-left:2px solid var(--rule);padding:4px 10px;margin:6px 0}
.turn .role{color:var(--olive);text-transform:uppercase;font-size:10px;letter-spacing:.08em}
.back{color:var(--copper);cursor:pointer;margin-bottom:14px;display:inline-block}
.note{color:var(--olive);max-width:70ch}
a.plain{color:var(--copper);cursor:pointer}
</style></head><body><div class="wrap">
<header><span class="mark">procgrep</span>
<nav>
  <span id="nav-query" onclick="show('query')">query</span>
  <span id="nav-permodel" onclick="show('permodel')">per-model</span>
  <span id="nav-curate" onclick="show('curate')">curate</span>
  <span id="nav-why" onclick="show('why')">why structural</span>
  <span class="dim"><a class="plain" href="https://github.com/hamidahoderinwale/procgrep" target="_blank">github ↗</a></span>
</nav></header>

<!-- QUERY (hero / default) -->
<div id="query">
<h1>Grep for agent traces.</h1>
<p class="sub">Ask a structural question about how an agent worked, like "did it edit five times with no test?" or "did it submit without testing?", and get an exact answer instantly, over real trajectory datasets. No model call, deterministic, free. <a class="plain" onclick="show('why')">Why not just ask an LLM? →</a></p>
<div class="dsbar" id="dsbar"></div>
<div class="qbar"><span class="mag">⌕</span>
  <input id="q" placeholder="atom pattern, e.g.  (edit ){5,}   or pick one below" autocomplete="off"></div>
<div class="tryline">try: <span id="trychips"></span> <span class="chip noisetog" onclick="document.body.classList.toggle('hide-noise')">hide think/other</span></div>
<div id="qres"></div>
</div>

<!-- CURATE -->
<div id="curate" class="hidden">
<h1>Trim to a budget, keep the variety.</h1>
<p class="sub">When you shrink a trajectory set to a fixed budget, the common move is to keep the shortest runs. But short traces are the least diverse: you throw away procedural variety. procgrep selects for diversity instead, preserving far more of the dataset's distinct procedures at the same budget.</p>
<div class="dsbar" id="dsbar-c"></div>
<div id="curbody"></div>
</div>

<!-- PER-MODEL -->
<div id="permodel" class="hidden">
<h1>How each model actually behaves.</h1>
<p class="sub">The same dataset, broken down by the model that produced each trajectory: how long its runs are, how much it repeats itself, and the mix of actions it favours.</p>
<div class="dsbar" id="dsbar-p"></div>
<div id="pmbody"></div>
</div>

<!-- WHY STRUCTURAL -->
<div id="why" class="hidden"></div>

<!-- TRACE DETAIL (on demand) -->
<div id="tr" class="hidden"></div>

<!-- BROWSE (ecosystem catalog, demoted) -->
<div id="browse" class="hidden">
<span class="back" onclick="show('query')">← query</span>
<h1 style="font-size:23px">The agent-trace ecosystem.</h1>
<p class="sub">Every public trajectory dataset procgrep discovered on Hugging Face, and what it can parse today. Most non-parseable entries aren't failures. They're benchmarks (task definitions, no trajectories), corpora, or out-of-scope domains.</p>
<div class="cards" id="findings"></div>
<div class="eyebrow">benchmarks (click to filter)</div>
<div class="bstrip" id="benchstrip"></div>
<div class="eyebrow">the index</div>
<div class="controls">
  <input id="bq" placeholder="filter datasets…" oninput="ST.q=this.value;ST.page=1;renderBrowse()" style="flex:1;min-width:180px">
  <label class="gb"><input type="checkbox" onchange="ST.group=this.checked;renderBrowse()" style="flex:none"> group by author</label>
  <span id="filters" class="dim"></span>
</div>
<table id="index"><thead><tr>
<th data-k="id">dataset</th><th>status</th><th data-k="downloads">downloads</th>
<th data-k="likes">likes</th><th data-k="last_modified">updated</th></tr></thead><tbody></tbody></table>
<button class="more hidden" id="more" onclick="ST.page++;renderBrowse()">view more</button>
<div class="foot" id="foot"></div>
</div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const CAT=D.catalog, DS=CAT.datasets, PROF=D.profiles;
const ATOM={search_repo:'#585E53',read_file:'#5692E5',edit:'#CB4D20',create_file:'#3D7AD8',
run_test:'#20A380',submit:'#14110E',think:'#b7b1a7',localize:'#8C1040',delete_file:'#A03D18',error:'#B4184F',other:'#d9d4cc'};
const pct=x=>x==null?'n/a':(x*100).toFixed(0)+'%';
const fmt=n=>n>=1000?(n/1000).toFixed(n>=10000?0:1)+'k':String(n);
const LOGOS=D.logos||{};
const PROFIDS=Object.keys(PROF);
let CURDS=PROFIDS[0]||null;
function blogo(b){return b?(LOGOS[b]?`<img class="blogo" src="${LOGOS[b]}">`:`<span class="bmono">${b.replace(/[^A-Za-z0-9]/g,'').slice(0,2).toUpperCase()}</span>`):'';}

// ---- shared dataset picker ----
function dsPicker(targetView){
  const opts=PROFIDS.map(id=>`<option value="${id}" ${id===CURDS?'selected':''}>${id}</option>`).join('');
  const sel=`<select onchange="CURDS=this.value;show('${targetView}')">${opts}</select>`;
  return `dataset: ${sel} <span class="dim">· ${PROFIDS.length} profiled</span> · <a class="plain" onclick="show('browse')">browse all ${DS.length} on HF →</a>`;
}

// ---- action-mix helpers ----
function mixOf(entries){const c={};let n=0;entries.forEach(e=>e.atoms.forEach(a=>{c[a]=(c[a]||0)+1;n++;}));
  const m={};Object.keys(c).forEach(k=>m[k]=c[k]/n);return {mix:m,n};}
function mixbar(m){return '<span class="mix">'+Object.entries(m).sort((a,b)=>b[1]-a[1]).map(([a,v])=>
  `<span style="width:${(v*160).toFixed(1)}px;background:${ATOM[a]||'#ccc'}" title="${a} ${pct(v)}"></span>`).join('')+'</span>';}
// Signal skeleton: collapse runs of think/other into one gap marker; run-length the signal atoms.
function skeleton(atoms){const r=[];for(const a of atoms){const noise=(a==='think'||a==='other');const t=r[r.length-1];
  if(noise){if(t&&t.gap)t.n++;else r.push({gap:true,n:1});}
  else{if(t&&t.a===a)t.n++;else r.push({a,n:1});}}return r;}
function spineMini(atoms){const sk=skeleton(atoms);return '<span class="spinemini">'+sk.slice(0,90).map(it=>{
  if(it.gap)return `<i class="nz" style="width:${Math.min(3+it.n,14)}px;background:#d9d4cc" title="${it.n} think/other"></i>`;
  const w=Math.min(5+(it.n-1)*2,16);
  return `<i style="width:${w}px;background:${ATOM[it.a]||'#ccc'}" title="${it.a}${it.n>1?' ×'+it.n:''}"></i>`;}).join('')+
  (sk.length>90?`<span class="dim" style="font-size:9px;margin-left:3px">+${sk.length-90}</span>`:'')+'</span>';}
function legend(m){return '<div class="legend">'+Object.entries(m).sort((a,b)=>b[1]-a[1]).filter(([,v])=>v>0.01).map(([a,v])=>
  `<span><i style="background:${ATOM[a]||'#ccc'}"></i>${a} ${pct(v)}</span>`).join('')+'</div>';}

// ---- sample structural queries ----
const QUERIES=[
 {id:'streak',label:'edit-streak ≥5',f:a=>/(?:edit ){5,}/.test(a.join(' ')+' ')},
 {id:'nosub',label:'submitted without testing',f:a=>a.includes('submit')&&!a.includes('run_test')},
 {id:'td',label:'test-driven',f:a=>{const r=a.indexOf('run_test'),e=a.indexOf('edit');return r>=0&&(e<0||r<e);}},
 {id:'stuck',label:'stuck reading',f:a=>/(?:read_file (?:think )?){4,}/.test(a.join(' ')+' ')},
 {id:'loop',label:'canonical resolve loop',f:a=>/search_repo read_file edit run_test/.test(a.join(' '))},
 {id:'nosearch',label:'never searched the repo',f:a=>!a.includes('search_repo')},
 {id:'repro',label:'wrote a repro script',f:a=>{const c=a.indexOf('create_file');return c>=0&&a.lastIndexOf('submit')>c;}},
 {id:'recover',label:'recovered from an error',f:a=>/error edit/.test(a.join(' '))},
];

// ---- QUERY view ----
function queryView(){
  document.getElementById('dsbar').innerHTML=dsPicker('query');
  const pool=(PROF[CURDS]||{}).atoms_pool||[];
  const counts=QUERIES.map(q=>[q,pool.filter(x=>q.f(x.atoms)).length]);
  document.getElementById('trychips').innerHTML=counts.map(([q,n])=>
    `<span class="chip ${n===0?'un':''}" onclick="runQ(null,'${q.id}')">${q.label} <span class="dim">${n}</span></span>`).join('');
  // auto-run the highest-hit sample so the hero always lands on a real result
  const top=counts.slice().sort((a,b)=>b[1]-a[1])[0];
  if(top&&top[1]>0&&!document.getElementById('q').value)runQ(null,top[0].id);
}
let _qt;
function onType(v){clearTimeout(_qt);_qt=setTimeout(()=>runQ(v),120);}
function runQ(custom,id){
  if(!CURDS||!PROF[CURDS]){document.getElementById('qres').innerHTML='<span class="dim">no profiled dataset loaded</span>';return;}
  const pool=PROF[CURDS].atoms_pool||[]; let f,label;
  const box=document.getElementById('q');
  if(id){const q=QUERIES.find(x=>x.id===id);f=q.f;label=q.label;if(box)box.value=q.id==='streak'?'(edit ){5,}':'';}
  else if(custom){let rx;try{rx=new RegExp(custom);}catch(e){document.getElementById('qres').innerHTML='<span class="dim">…keep typing, invalid pattern</span>';return;}
    f=a=>rx.test(a.join(' ')+' ');label='/'+custom+'/';}
  else{document.getElementById('qres').innerHTML='<span class="dim">type an action pattern, or pick a sample query above</span>';return;}
  const t=performance.now();const hits=pool.filter(x=>f(x.atoms));const ms=performance.now()-t;
  // per-model match rates
  const bym={}; pool.forEach(x=>{(bym[x.model]=bym[x.model]||{n:0,h:0}).n++;});
  hits.forEach(x=>{bym[x.model].h++;});
  const top=Object.entries(bym).map(([m,c])=>[m,c.h/c.n]).sort((a,b)=>b[1]-a[1]);
  const affected=top.length&&top[0][1]>0?`${prettyModel(top[0][0])} most affected (${pct(top[0][1])})`:'no model affected';
  const modelBars=top.map(([m,r])=>
    `<div class="rrow"><span class="rlab">${prettyModel(m)}</span>
      <span class="barbg"><span class="fill" style="width:${(r*120).toFixed(0)}px;background:var(--copper)"></span></span>
      <span class="rval">${pct(r)}</span></div>`).join('');
  // matched vs all action mix
  const allMix=mixOf(pool), hitMix=hits.length?mixOf(hits):{mix:{},n:0};
  document.getElementById('qres').innerHTML=
   `<div style="font-size:15px;margin:4px 0"><b>${hits.length}</b> / ${pool.length} traces match <b>${label}</b></div>
    <div class="dim" style="margin-bottom:12px"><span class="speed">scanned in ${ms.toFixed(1)} ms, no model call, $0</span> · ${affected}</div>
    <div class="eyebrow">which models</div>${modelBars||'<span class="dim">none</span>'}
    <div class="eyebrow">action mix · matched vs. all</div>
    <div class="rrow"><span class="rlab">matched</span>${hits.length?mixbar(hitMix.mix):'<span class="dim">no matches</span>'}</div>
    <div class="rrow"><span class="rlab">all traces</span>${mixbar(allMix.mix)}</div>
    ${hits.length?legend(hitMix.mix):''}
    <div class="eyebrow">matching traces (click to open)</div>
    ${hits.slice(0,40).map(h=>`<div class="qhit" onclick="poolTrace('${h.trace_id}')"><span class="qhmodel">${prettyModel(h.model)}</span> <span class="dim">${h.atoms.length} steps</span> ${spineMini(h.atoms)}</div>`).join('')}
    ${hits.length>40?`<div class="dim" style="margin-top:6px">+ ${hits.length-40} more</div>`:''}`;
}
function poolTrace(tid){const p=PROF[CURDS];const s=(p.samples||[]).find(x=>x.trace_id===tid);
  if(s){trace(CURDS,p.samples.indexOf(s));return;}
  const h=(p.atoms_pool||[]).find(x=>x.trace_id===tid);if(!h)return;show('tr');
  const spine=skeleton(h.atoms).map(it=>it.gap?`<span class="gap" title="${it.n} think/other steps">···${it.n}···</span>`:`<span class="atom" style="background:${ATOM[it.a]||'#ccc'}">${it.a}${it.n>1?' ×'+it.n:''}</span>`).join('');
  document.getElementById('tr').innerHTML=`<span class="back" onclick="show('query')">← query</span>
    <h1 style="font-size:19px">${h.model} · ${h.atoms.length} steps</h1>
    <div class="eyebrow">procedural spine</div><div class="spine">${spine}</div>
    <div class="dim">conversation not sampled for this trace</div>`;}

// ---- CURATE view ----
function curateView(){
  document.getElementById('dsbar-c').innerHTML=dsPicker('curate');
  const p=PROF[CURDS]; const r=p.redundancy;
  const s=r.coverage_shortest, d=r.coverage_diverse;
  document.getElementById('curbody').innerHTML=
   `<div class="eyebrow">${p.dataset.split('/').pop()} · ${p.n_traces} traces</div>
    <p class="note">Shrink the set to a fixed budget, <b>the same number of traces</b>, chosen two ways.
    <b>Procedural variety</b> = the share of the dataset's distinct procedures (recurring action
    sub-sequences) that still appear in the kept subset.</p>
    <div class="rrow" style="margin:12px 0"><span class="rlab">keep shortest</span>
      <span class="barbg" style="width:280px"><span class="fill" style="width:${s*280}px;background:var(--olive)"></span></span>
      <span class="rval" style="width:auto">${pct(s)} of procedures</span></div>
    <div class="rrow" style="margin:12px 0"><span class="rlab">keep diverse</span>
      <span class="barbg" style="width:280px"><span class="fill" style="width:${d*280}px;background:var(--copper)"></span></span>
      <span class="rval" style="width:auto">${pct(d)} ← procgrep</span></div>
    <p class="note" style="margin-top:18px">Same budget, but diversity-selection keeps <b>${pct(d)}</b> of the
    distinct procedures against <b>${pct(s)}</b> for keep-shortest, a <b>${Math.round((d-s)*100)}-point</b> gain at no
    extra cost. Short runs cluster in the redundant core (${pct(r.exact_dup_rate)} of traces are exact duplicates,
    ${pct(r.near_dup_rate)} near-duplicates), so keeping the shortest over-samples sameness.</p>`;
}

// ---- PER-MODEL view ----
function permodelView(){
  document.getElementById('dsbar-p').innerHTML=dsPicker('permodel');
  const p=PROF[CURDS];
  const rows=Object.entries(p.by_model).map(([m,s])=>
    `<tr><td>${prettyModel(m)}</td><td>${s.n}</td><td>${s.median_len}</td><td>${pct(s.exact_dup_rate)}</td><td>${mixbar(s.action_mix)}</td></tr>`).join('');
  document.getElementById('pmbody').innerHTML=
   `<div class="eyebrow">${p.dataset.split('/').pop()} · ${p.adapter} · ${p.n_models} models</div>
    <table><thead><tr><th>model</th><th>n</th><th>median len</th><th>exact-dup</th><th>action mix</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="eyebrow">sampled traces (click to open the conversation)</div>
    <div>${(p.samples||[]).map((s,i)=>`<span class="chip" onclick="trace('${CURDS}',${i})">trace ${i+1} · ${prettyModel(s.model)} · ${s.atoms.length} steps</span>`).join('')}</div>`;
}
function trace(id,i){const s=PROF[id].samples[i];show('tr');
  const spine=skeleton(s.atoms).map(it=>it.gap?`<span class="gap" title="${it.n} think/other steps">···${it.n}···</span>`:`<span class="atom" style="background:${ATOM[it.a]||'#ccc'}">${it.a}${it.n>1?' ×'+it.n:''}</span>`).join('');
  const turns=(s.turns||[]).map(t=>{const body=t.tools?('⚙ '+t.tools.join(', ')):(t.text||'');
    return `<div class="turn"><span class="role">${t.role}</span><br>${(body+'').replace(/</g,'&lt;')}</div>`;}).join('');
  document.getElementById('tr').innerHTML=
   `<span class="back" onclick="show('permodel')">← per-model</span><h1 style="font-size:19px">${s.model} · ${s.atoms.length} steps</h1>
    <div class="eyebrow">procedural spine</div><div class="spine">${spine}</div><div class="eyebrow">conversation</div>${turns}`;}

// ---- WHY STRUCTURAL view ----
// Plain-language names for the structural predicates + the model/task prettifiers.
const PRED_LABEL={edit_streak_5:'edited 5+ times in a row',submitted_without_test:'submitted without testing',
  tested_before_first_edit:'tested before its first edit',read_streak_4:'read 4+ files in a row',
  never_searched:'never searched the repo'};
function prettyModel(a){let m=String(a);
  // provider/model judge slugs, e.g. anthropic/claude-sonnet-4-6
  if(/^(openai|anthropic|deepseek|google|meta)\//i.test(m)){
    m=m.replace(/^[^/]+\//,'');                         // strip provider prefix
    const claudeGpt=/^(claude|gpt)/i.test(m);
    if(claudeGpt)m=m.replace(/(\d)-(\d)$/,'$1.$2');     // 4-6 -> 4.6 only for claude/gpt
    if(/^claude/i.test(m)){
      m='Claude '+m.replace(/^claude-?/i,'').split('-').map(w=>
        /^\d/.test(w)?w:w.charAt(0).toUpperCase()+w.slice(1)).join(' ');
    }else if(/^deepseek/i.test(m)){
      m='DeepSeek '+m.replace(/^deepseek-?/i,'').split('-').map(w=>
        w.charAt(0).toUpperCase()+w.slice(1)).join(' ');
    }else if(/^gpt/i.test(m)){
      // keep the GPT-4o/GPT-4 hyphen, flatten the rest, e.g. gpt-4o-mini -> GPT-4o mini
      const rest=m.replace(/^gpt-?(4o|4)-?/i,'').replace(/-/g,' ').trim();
      m='GPT-'+(/4o/i.test(m)?'4o':'4')+(rest?' '+rest:'');
    }else{
      m=m.replace(/-/g,' ');
    }
    return m.trim()||String(a);
  }
  // agent-trace ids, e.g. swe-agent-llama-70b
  m=m.replace(/^(swe-agent|agentless|moatless|dars|mini-swe-agent)[-+]/i,'');
  m=m.replace(/llama-?(\d+)b/i,(_,n)=>'Llama '+n+'B').replace(/\bgpt-?4o\b/i,'GPT-4o').replace(/\bgpt-?4\b/i,'GPT-4')
     .replace(/claude-?([\d.]+)([a-z-]*)/i,(_,v,s)=>'Claude '+v+(s?' '+s.replace(/-/g,' ').trim():''))
     .replace(/deepseek-?(v?\d+|r\d+)/i,(_,v)=>'DeepSeek '+v.toUpperCase());
  return (m.replace(/-/g,' ').trim())||String(a);}
function f1col(v){if(v==null)return '#efeae2';const t=Math.max(0,Math.min(1,v)),a=[247,238,232],b=[32,163,128];
  return 'rgb('+a.map((x,k)=>Math.round(x+(b[k]-x)*t)).join(',')+')';}  // pale -> teal as F1 rises
function whyView(){const e=D.experiment;const el=document.getElementById('why');
  if(!e){el.innerHTML='<span class="back" onclick="show(\'query\')">← query</span><p class="sub">experiment not loaded.</p>';return;}
  const judges=Object.entries(e.pareto||{}).filter(([k])=>k!=='procgrep');
  const meanLat=judges.length?judges.reduce((s,[,v])=>s+(v.mean_latency_s||0),0)/judges.length:1;
  const ratio=Math.round(meanLat*1e6/e.procgrep_us_per_decision);
  const bestF1=judges.length?Math.max(...judges.map(([,v])=>v.mean_f1||0)):0;
  const struct=Object.entries(e.predicates).filter(([k,v])=>v.kind==='structural');
  const models=Object.keys(struct[0][1].judges);
  const head=`<th>behavioural question</th>${models.map(m=>`<th>${prettyModel(m)}</th>`).join('')}<th>judge agreement κ</th>`;
  const body=struct.map(([k,v])=>`<tr><td>${PRED_LABEL[k]||k}</td>${models.map(m=>{const f=v.judges[m].f1;
    return `<td style="background:${f1col(f)};text-align:center">${f==null?'n/a':f.toFixed(2)}</td>`;}).join('')}<td style="text-align:center">${v.pairwise_cohen_kappa==null?'n/a':v.pairwise_cohen_kappa}</td></tr>`).join('');
  const fz=Object.entries(e.predicates).find(([k,v])=>v.kind==='fuzzy');
  el.innerHTML=
   `<span class="back" onclick="show('query')">← query</span>
    <h1>Why not just ask an LLM?</h1>
    <p class="sub">Every behavioural question in the query tab can be asked two ways: a deterministic structural query over the action spine, or an LLM judge reading the trace. We ran both over the same trajectories, including a strong judge.</p>
    <div class="cards">
      <div class="card"><div class="dim">speed</div><div class="big">${ratio.toLocaleString()}×</div>
        <div class="dim">procgrep ${e.procgrep_us_per_decision} µs vs LLM ${meanLat.toFixed(1)}s per decision, and $0 vs API cost</div></div>
      <div class="card"><div class="dim">accuracy, mean F1 where procgrep = 1.0</div><div class="big">≤ ${bestF1.toFixed(2)}</div>
        <div class="dim">best judge ${bestF1.toFixed(2)}; F1 = 0 on both counting questions for every judge, because LLMs cannot count action streaks in a trace</div></div>
      <div class="card"><div class="dim">reliability</div><div class="big">κ ≈ 0</div>
        <div class="dim">judges barely agree with each other, on structural facts and fuzzy ones alike</div></div>
    </div>
    <div class="eyebrow">how often each LLM judge matches the exact structural answer</div>
    <p class="note" style="margin:0 0 10px">Each cell is a judge's F1 against the exact answer: 1.00 is perfect, 0 is useless, chance is about 0.5. Greener is higher. procgrep is 1.00 by construction. κ in the last column is how much the judges agree with one another (near 0 means barely).</p>
    <table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
    <p class="note">On the one fuzzy question, <b>${fz?(PRED_LABEL[fz[0]]||fz[0]):''}</b>, the judges agree at κ = ${fz?fz[1].pairwise_cohen_kappa:'n/a'}. procgrep answers all ${struct.length} questions over ${e.corpus} traces in ${e.procgrep_ms_full} ms; the LLM route is ${ratio.toLocaleString()}× slower, costs per call, and is no more reliable (best mean F1 ${bestF1.toFixed(2)}). It took ${e.totals.judge_calls} judge calls and ${e.totals.total_tokens.toLocaleString()} tokens.</p>`;}

// ---- BROWSE view (ecosystem catalog, demoted) ----
const ST={q:'',sort:'downloads',dir:-1,group:false,page:1,size:30,fmtFilter:null,statusFilter:null,benchFilter:null,collapsed:new Set(),open:new Set()};
// Honest classification: never surface a raw exception to the reader.
function classify(r){
  if(r.adapter)return {key:'parseable',label:r.adapter,cls:'fmt',why:`parseable, format <b>${r.adapter}</b> (confidence ${r.confidence})`};
  if(r.candidate)return {key:'format-gap',label:'format gap',cls:'gap',why:'a trajectory dataset procgrep does not parse yet, on the roadmap'};
  if(r.out_of_scope)return {key:'out-of-scope',label:'out of scope',cls:'un',why:'a non-tool domain (vision, robotics, math) with no discrete code actions to canonicalize'};
  const e=(r.error||'');
  if(/gated/i.test(e))return {key:'gated',label:'gated',cls:'un',why:'a gated dataset on the Hub, needs authentication to introspect'};
  if(/Config name is missing/i.test(e))return {key:'multi-config',label:'multi-config',cls:'un',why:'a multi-config dataset, a config must be selected before introspection'};
  if(r.benchmark)return {key:'benchmark',label:'benchmark',cls:'un',why:`task definitions for the ${r.benchmark} benchmark, not agent trajectories`};
  return {key:'not-a-trace',label:'not a trace',cls:'un',why:'no conversation or trajectory column, a corpus or instruction dataset, not agent traces'};
}
function dupOf(r){const p=PROF[r.id];return p?p.redundancy.exact_dup_rate:null;}
function findings(){
  const cand=DS.filter(r=>r.candidate), sup=DS.filter(r=>r.supported);
  const notTrace=DS.filter(r=>classify(r).key==='not-a-trace'||classify(r).key==='benchmark').length;
  const oos=DS.filter(r=>classify(r).key==='out-of-scope').length;
  const byA={}; sup.forEach(r=>byA[r.adapter]=(byA[r.adapter]||0)+1);
  const fmtChips=Object.entries(byA).sort((a,b)=>b[1]-a[1]).map(([a,n])=>
    `<span class="chip fmt ${ST.fmtFilter===a?'on':''}" onclick="ST.fmtFilter=ST.fmtFilter===\'${a}\'?null:\'${a}\';ST.page=1;renderBrowse()">${a} ${n}</span>`).join('');
  document.getElementById('findings').innerHTML=
   `<div class="card"><div class="dim">coverage, parseable trace datasets</div><div class="big">${sup.length}/${cand.length}</div>
      <div class="dim">${notTrace} benchmarks/corpora &amp; ${oos} out-of-scope excluded · ${CAT.n_discovered} discovered</div></div>
    <div class="card"><div class="dim">formats (click to filter)</div><div style="margin-top:8px">${fmtChips}</div></div>
    <div class="card"><div class="dim">what's excluded, honestly</div>
      <div class="dim" style="margin-top:8px">non-parseable rows are benchmarks (task definitions), corpora, gated, or out-of-scope domains, not parse failures on real traces</div></div>`;
  const benches={}; DS.forEach(r=>{if(r.benchmark)(benches[r.benchmark]=benches[r.benchmark]||[]).push(r);});
  const bs=document.getElementById('benchstrip');
  if(bs)bs.innerHTML=Object.entries(benches).sort((a,b)=>b[1].length-a[1].length).map(([b,rs])=>{
    const s=rs.filter(r=>r.supported).length;
    return `<span class="bcard ${ST.benchFilter===b?'on':''}" onclick="ST.benchFilter=ST.benchFilter===\'${b}\'?null:\'${b}\';ST.page=1;renderBrowse()">${blogo(b)} ${b} <span class="dim">${s}/${rs.length}</span></span>`;}).join('');
}
function rowsFiltered(){
  let rows=DS.filter(r=>r.id.toLowerCase().includes(ST.q.toLowerCase()));
  if(ST.fmtFilter)rows=rows.filter(r=>r.adapter===ST.fmtFilter);
  if(ST.benchFilter)rows=rows.filter(r=>r.benchmark===ST.benchFilter);
  if(ST.statusFilter)rows=rows.filter(r=>classify(r).key===ST.statusFilter);
  const key=ST.sort;
  rows.sort((a,b)=>{let va,vb;
    if(key==='dup'){va=dupOf(a)??-1;vb=dupOf(b)??-1;}
    else{va=a[key]??'';vb=b[key]??'';}
    return (va<vb?-1:va>vb?1:0)*ST.dir;});
  return rows;
}
function rowHTML(r){
  const c=classify(r);const p=PROF[r.id];
  let h=`<tr class="clk"><td class="idcell" onclick="toggleRow('${r.id}')">${blogo(r.benchmark)} ${r.id}</td>
    <td><span class="chip ${c.cls}">${c.label}</span></td>
    <td>${fmt(r.downloads)}</td><td>${r.likes||0}</td><td class="dim">${r.last_modified||'n/a'}</td></tr>`;
  if(ST.open.has(r.id)){
    const prof=p?` · <a class="plain" onclick="CURDS='${r.id}';show('query')">query it →</a>`:'';
    h+=`<tr><td colspan="5" class="detail"><a href="https://huggingface.co/datasets/${r.id}" target="_blank">huggingface.co/datasets/${r.id} ↗</a> · ${c.why}${prof}</td></tr>`;}
  return h;
}
function toggleRow(id){ST.open.has(id)?ST.open.delete(id):ST.open.add(id);renderBrowse();}
function toggleGroup(au){ST.collapsed.has(au)?ST.collapsed.delete(au):ST.collapsed.add(au);renderBrowse();}
function renderBrowse(){
  findings();
  const rows=rowsFiltered();
  const tb=document.querySelector('#index tbody');
  if(ST.group){
    const g={}; rows.forEach(r=>(g[r.author]=g[r.author]||[]).push(r));
    tb.innerHTML=Object.entries(g).sort((a,b)=>b[1].length-a[1].length).map(([au,rs])=>{
      const open=!ST.collapsed.has(au);
      const sup=rs.filter(r=>r.supported).length;
      const hdr=`<tr class="grouphdr" onclick="toggleGroup('${au}')"><td colspan="5">${open?'▾':'▸'} ${au} `+
                `<span class="dim">· ${rs.length} datasets · ${sup} parseable</span></td></tr>`;
      return hdr + (open ? rs.map(rowHTML).join('') : '');
    }).join('');
    document.getElementById('more').classList.add('hidden');
  }else{
    const shown=rows.slice(0,ST.page*ST.size);
    tb.innerHTML=shown.map(rowHTML).join('');
    document.getElementById('more').classList.toggle('hidden',shown.length>=rows.length);
  }
  document.getElementById('filters').innerHTML=
    ['parseable','format-gap','benchmark','not-a-trace','out-of-scope','gated'].map(s=>`<span class="chip ${ST.statusFilter===s?'on':''}" onclick="ST.statusFilter=ST.statusFilter===\'${s}\'?null:\'${s}\';ST.page=1;renderBrowse()">${s}</span>`).join('');
  document.getElementById('foot').textContent=
    `catalog generated ${CAT.generated||'n/a'} · ${CAT.n_discovered} datasets discovered · ${DS.length} sniffed · showing ${rows.length} after filters`;
}
document.querySelectorAll('#index th[data-k]').forEach(th=>th.onclick=()=>{
  const k=th.dataset.k;ST.dir=(k===ST.sort)?-ST.dir:-1;ST.sort=k;renderBrowse();});

// ---- view router ----
const VIEWS=['query','curate','permodel','why','tr','browse'];
const NAVOF={query:'nav-query',curate:'nav-curate',permodel:'nav-permodel',why:'nav-why'};
function show(id){
  VIEWS.forEach(x=>document.getElementById(x).classList.toggle('hidden',x!==id));
  ['nav-query','nav-curate','nav-permodel','nav-why'].forEach(n=>document.getElementById(n).classList.remove('act'));
  if(NAVOF[id])document.getElementById(NAVOF[id]).classList.add('act');
  else if(id==='tr'||id==='browse')document.getElementById('nav-query').classList.add('act');
  if(id==='query')queryView();
  if(id==='curate')curateView();
  if(id==='permodel')permodelView();
  if(id==='why')whyView();
  if(id==='browse')renderBrowse();
  window.scrollTo(0,0);
}
document.getElementById('q').addEventListener('input',e=>onType(e.target.value));

// boot: query-first; queryView() auto-runs the highest-hit sample
show('query');
// deep links
const _h=location.hash.replace('#','');
if(['why','browse','curate','permodel','query'].includes(_h))show(_h);
else if(location.hash.startsWith('#ds=')){const [id,q]=location.hash.slice(4).split('&q=');
  const did=decodeURIComponent(id);if(PROF[did]){CURDS=did;show('query');if(q)runQ(null,q);}}
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--profiles", nargs="*", default=[])
    ap.add_argument("--experiment", default=None, help="query_vs_llm_full.json for the 'why structural' page")
    ap.add_argument("--logos", default=None, help="benchmark->dataURI logo map JSON")
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
    logos = json.loads(Path(args.logos).read_text()) if args.logos else {}

    data = json.dumps({"catalog": catalog, "profiles": profiles, "experiment": experiment, "logos": logos})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", data))
    print(f"wrote {out}  ({len(catalog['datasets'])} datasets, {len(profiles)} profiles)")


if __name__ == "__main__":
    main()
