// ProcGrep explorer frontend. Talks to the FastAPI backend (/datasets, /query);
// no embedded data, so it queries whole datasets server-side rather than a sample.
const SAMPLES = [
  { label: "edit-streak ≥5", pat: "(edit ){5,}" },
  { label: "submitted without testing", pat: "^(?:(?!run_test).)*submit" },
  { label: "never searched the repo", pat: "^(?:(?!search_repo).)*$" },
  { label: "stuck reading", pat: "(read_file (?:think )?){4,}" },
  { label: "canonical resolve loop", pat: "search_repo read_file edit run_test" },
  { label: "recovered from an error", pat: "error edit" },
];
const $ = (id) => document.getElementById(id);
function prettyModel(a){let m=String(a).replace(/^(swe-agent|agentless|moatless|dars|mini-swe-agent)[-+]/i,'');
  m=m.replace(/llama-?(\d+)b/i,(_,n)=>'Llama '+n+'B').replace(/\bgpt-?4o\b/i,'GPT-4o').replace(/\bgpt-?4\b/i,'GPT-4')
     .replace(/claude-?([\d.]+)([a-z-]*)/i,(_,v,sfx)=>'Claude '+v+(sfx?' '+sfx.replace(/-/g,' ').trim():''))
     .replace(/deepseek-?(v?\d+|r\d+)/i,(_,v)=>'DeepSeek '+v.toUpperCase());
  return (m.replace(/-/g,' ').trim())||String(a);}
function prettyTask(t){if(!t)return '';const m=String(t).match(/^(.+?)__(.+?)-(\d+)$/);return m?`${m[1]}/${m[2]} #${m[3]}`:String(t);}
let HIDE_NOISE = false;
let DSETS = [];                 // store-backed dataset ids (shared by query + compare)
const CMP = { axis: "agent", ds: null, left: null, right: null, data: null };

async function loadDatasets() {
  const d = await (await fetch("datasets")).json();
  DSETS = d.suggested;
  CMP.ds = DSETS[0];
  $("ds").innerHTML = d.suggested.map((s) => `<option>${s}</option>`).join("");
  $("trychips").innerHTML = SAMPLES.map((s, i) => `<span class="chip" data-i="${i}">${s.label}</span>`).join("");
}

// Collapse runs of think/other into a gap marker; run-length the signal atoms.
function spine(atoms, color) {
  const runs = [];
  for (const a of atoms) {
    const noise = a === "think" || a === "other";
    const t = runs[runs.length - 1];
    if (noise) { if (t && t.gap) t.n++; else runs.push({ gap: true, n: 1 }); }
    else { if (t && t.a === a) t.n++; else runs.push({ a, n: 1 }); }
  }
  return '<span class="spine">' + runs.slice(0, 90).map((it) => it.gap
    ? `<i class="nz" title="${it.n} think/other" style="width:${Math.min(3 + it.n, 14)}px;background:#d9d4cc"></i>`
    : `<i title="${it.a}${it.n > 1 ? " ×" + it.n : ""}" style="width:${Math.min(5 + (it.n - 1) * 2, 16)}px;background:${color[it.a] || "#ccc"}"></i>`
  ).join("") + "</span>";
}

function mixbar(mix, color) {
  return '<span class="mix">' + Object.entries(mix).sort((a, b) => b[1] - a[1]).map(
    ([a, v]) => `<span title="${a} ${(v * 100).toFixed(0)}%" style="width:${(v * 200).toFixed(1)}px;background:${color[a] || "#ccc"}"></span>`
  ).join("") + "</span>";
}

async function run(pattern) {
  const ds = $("ds").value;
  $("res").innerHTML = '<span class="dim">scanning the full dataset on the server…</span>';
  $("q").value = pattern;
  let r;
  try {
    r = await (await fetch("query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset: ds, pattern }),
    })).json();
  } catch (e) {
    $("res").innerHTML = `<span class="err">request failed: ${e}</span>`;
    return;
  }
  if (r.error) { $("res").innerHTML = `<span class="err">${r.error}</span>`; return; }
  $("dsmeta").textContent = `· ${r.n_traces} traces${r.truncated ? " (capped)" : ""}`;
  const col = r.atom_color;
  const top = r.by_model[0];
  const s = r.stats || {};
  const statline = s.n_models
    ? `<div class="dim" style="margin-bottom:14px">diversity ${s.diversity_bits} bits · median ${s.median_len} steps · CoT ${s.median_cot} think-steps · ${(s.exact_dup_rate * 100).toFixed(0)}% exact-duplicate · ${s.n_models} models</div>`
    : "";
  $("res").innerHTML =
    `<div style="font-size:15px;margin:4px 0"><b>${r.n_hits}</b> / ${r.n_traces} traces match <b>/${r.pattern}/</b></div>
     <div class="dim" style="margin-bottom:6px"><span class="speed">scanned in ${r.elapsed_ms} ms, no model call</span>${top ? ` · ${prettyModel(top.model)} most affected at ${(top.rate * 100).toFixed(0)}%` : ""}</div>
     ${statline}
     <div class="eyebrow">which models</div>
     ${r.by_model.map((m) => `<div class="rrow"><span class="rlab">${prettyModel(m.model)}</span><span class="barbg"><span class="fill" style="width:${(m.rate * 160).toFixed(0)}px"></span></span><span>${(m.rate * 100).toFixed(0)}%</span></div>`).join("")}
     <div class="eyebrow">action mix · matched vs. all</div>
     <div class="rrow"><span class="rlab">matched</span>${r.n_hits ? mixbar(r.mix_hits, col) : '<span class="dim">no matches</span>'}</div>
     <div class="rrow"><span class="rlab">all traces</span>${mixbar(r.mix_all, col)}</div>
     <div class="eyebrow">matching traces</div>
     ${r.hits.map((h) => `<div class="qhit"><span class="hit-model">${prettyModel(h.model)}</span>${h.task ? `<span class="hit-task dim" title="${prettyTask(h.task)}">${prettyTask(h.task)}</span>` : ""}<span class="dim">${h.atoms.length} steps</span>${spine(h.atoms, col)}</div>`).join("")}
     ${r.n_hits > r.hits.length ? `<div class="note">showing ${r.hits.length} of ${r.n_hits} matches</div>` : ""}`;
}

document.addEventListener("click", (e) => {
  const c = e.target.closest(".chip[data-i]");
  if (c) run(SAMPLES[+c.dataset.i].pat);
  if (e.target.id === "noisetog") {
    HIDE_NOISE = !HIDE_NOISE;
    document.body.classList.toggle("hide-noise", HIDE_NOISE);
  }
});
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") run($("q").value); });
$("ds").addEventListener("change", () => run($("q").value || SAMPLES[0].pat));

// ── comparator ────────────────────────────────────────────────────────────
// A trail-as-thread: collapse think/other runs into fold lines, render signal
// atoms as labeled colored steps. The same skeleton the spine() bars draw,
// expanded into a readable column. This is how we show a "conversation":
// procgrep keeps the action structure, not the raw text.
function threadHTML(atoms, color) {
  const out = [];
  let i = 0;
  while (i < atoms.length) {
    const a = atoms[i];
    if (a === "think" || a === "other") {
      let n = 0;
      while (i < atoms.length && (atoms[i] === "think" || atoms[i] === "other")) { n++; i++; }
      out.push(`<div class="tstep fold">⋯ ${n} reasoning ${n > 1 ? "steps" : "step"}</div>`);
    } else {
      let n = 1;
      while (i + n < atoms.length && atoms[i + n] === a) n++;
      out.push(`<div class="tstep"><span class="ti">${i + 1}</span><span class="dot" style="background:${color[a] || "#ccc"}"></span>${a}${n > 1 ? " ×" + n : ""}</div>`);
      i += n;
    }
  }
  return out.join("");
}

// A contiguous trajectory barcode: one cell per atom, reasoning pale and
// actions vivid, so trail length is proportional and behavior reads as color
// density. Used by the comparator (vs the collapsing spine() the query uses).
function barcode(atoms, color) {
  return '<span class="bc">' + atoms.map((a) => {
    const noise = a === "think" || a === "other";
    return `<i title="${a}" style="background:${noise ? "#e6e1d8" : (color[a] || "#ccc")}"></i>`;
  }).join("") + "</span>";
}

function openTrace(side, i) {
  const r = CMP.data; if (!r) return;
  const t = r[side].trails[i];
  const ds = CMP.axis === "eval" ? (side === "left" ? CMP.left : CMP.right) : CMP.ds;
  const src = `https://huggingface.co/datasets/${ds}`;
  const sheet = document.createElement("div");
  sheet.className = "sheet";
  sheet.onclick = (e) => { if (e.target === sheet) sheet.remove(); };
  sheet.innerHTML =
    `<div class="card"><span class="x" onclick="this.closest('.sheet').remove()">close ✕</span>
     <div style="font-size:15px"><b>${prettyModel(t.model)}</b>${t.task ? ` · ${prettyTask(t.task)}` : ""}</div>
     <div class="dim" style="margin:2px 0 4px">${t.steps} steps · structural trail, no raw text stored</div>
     <div style="margin:6px 0 12px">${barcode(t.atoms, r.atom_color)}</div>
     <div class="thread">${threadHTML(t.atoms, r.atom_color)}</div>
     <div class="note" style="margin-top:12px"><a href="${src}" target="_blank" rel="noopener">view full trace at source ↗</a> · ${ds}</div>
     </div>`;
  document.body.appendChild(sheet);
}

function trailStack(side) {
  const r = CMP.data, col = r[side], c = r.atom_color;
  const s = col.stats;
  return `<div class="cmpcol">
    <div class="cmphead">${prettyModel(col.label)}</div>
    <div class="cmpstat">${col.n.toLocaleString()} traces · median ${s.median_len} steps · ${s.median_cot} reasoning · diversity ${s.diversity_bits} bits</div>
    ${col.trails.map((t, i) => `<div class="trow" onclick="openTrace('${side}',${i})"><span class="tlen">${t.steps} st</span>${barcode(t.atoms, c)}</div>`).join("")}</div>`;
}

function diffStrip() {
  const r = CMP.data, c = r.atom_color, d = r.diff;
  const L = prettyModel(r.left.label), R = prettyModel(r.right.label);
  const procs = (d.procedures || []).slice(0, 7).map((p) => {
    const lean = p.log_odds >= 0 ? `<span class="lean l">${L} ◂</span>` : `<span class="lean r">▸ ${R}</span>`;
    return `<div class="proc">${lean}${barcode(p.atoms, c)}</div>`;
  }).join("");
  return `<div class="diffstrip">
    <div class="dnums">
      <div class="dnum"><div class="big">${d.jsd == null ? "—" : d.jsd}</div><div class="lab">action-mix JSD, 0 same to 1 disjoint</div></div>
      <div class="dnum"><div class="big">${d.len_delta > 0 ? "+" : ""}${d.len_delta}</div><div class="lab">median length, <span class="lead-l">${L}</span> minus <span class="lead-r">${R}</span></div></div>
      <div class="dnum"><div class="big">${d.cot_delta > 0 ? "+" : ""}${d.cot_delta}</div><div class="lab">reasoning steps, same direction</div></div>
    </div>
    <div class="eyebrow" style="margin:0 0 6px">procedures that most separate them</div>
    ${procs || '<span class="dim">no distinguishing procedures found</span>'}
    <div class="note">◂ leans <span class="lead-l">${L}</span> · leans <span class="lead-r">${R}</span> ▸ · think/other runs collapse to gaps</div>
  </div>`;
}

async function runCompare() {
  if (!CMP.left || !CMP.right) return;
  $("cmp-res").innerHTML = '<span class="dim">diffing the two groups on the server…</span>';
  let r;
  try {
    r = await (await fetch("compare", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ axis: CMP.axis, dataset: CMP.ds, left: CMP.left, right: CMP.right }),
    })).json();
  } catch (e) { $("cmp-res").innerHTML = `<span class="err">request failed: ${e}</span>`; return; }
  if (r.error) { $("cmp-res").innerHTML = `<span class="err">${r.error}</span>`; return; }
  CMP.data = r;
  $("cmp-res").innerHTML = `${diffStrip()}<div class="cmp">${trailStack("left")}${trailStack("right")}</div>`;
}

function opt(sel, list, val) {
  return `<select onchange="${sel}">${list.map((x) => `<option ${x === val ? "selected" : ""}>${x}</option>`).join("")}</select>`;
}

async function renderCmpControls() {
  const box = $("cmp-controls");
  if (CMP.axis === "eval") {
    if (!DSETS.includes(CMP.left) || CMP.left === CMP.right) { CMP.left = DSETS[0]; CMP.right = DSETS[1]; }
    box.innerHTML = `${opt("CMP.left=this.value;runCompare()", DSETS, CMP.left)}
      <span class="dim">vs</span>${opt("CMP.right=this.value;runCompare()", DSETS, CMP.right)}`;
  } else {
    box.innerHTML = `${opt("CMP.ds=this.value;renderCmpControls()", DSETS, CMP.ds)} <span class="dim">·</span> <span id="agentpick" class="dim">loading agents…</span>`;
    const g = await (await fetch(`groups?dataset=${encodeURIComponent(CMP.ds)}`)).json();
    const agents = (g.agents || []).map((a) => a.agent);
    if (agents.length < 2) { $("agentpick").innerHTML = '<span class="err">only one agent in this dataset; pick another or use by eval</span>'; return; }
    CMP.left = agents[0]; CMP.right = agents[1];
    $("agentpick").innerHTML = `${opt("CMP.left=this.value;runCompare()", agents, CMP.left)}
      <span class="dim">vs</span>${opt("CMP.right=this.value;runCompare()", agents, CMP.right)}`;
  }
  runCompare();
}

function setAxis(axis) {
  CMP.axis = axis;
  $("seg-agent").classList.toggle("act", axis === "agent");
  $("seg-eval").classList.toggle("act", axis === "eval");
  renderCmpControls();
}

let CMP_INIT = false;
function showView(name) {
  $("view-query").classList.toggle("hidden", name !== "query");
  $("view-compare").classList.toggle("hidden", name !== "compare");
  $("tab-query").classList.toggle("act", name === "query");
  $("tab-compare").classList.toggle("act", name === "compare");
  if (name === "compare" && !CMP_INIT) { CMP_INIT = true; setAxis("agent"); }
}

loadDatasets().then(() => {
  run(SAMPLES[0].pat);
  if ((location.hash || "").replace("#", "") === "compare") showView("compare");
});
