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
let HIDE_NOISE = false;

async function loadDatasets() {
  const d = await (await fetch("datasets")).json();
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
  $("res").innerHTML =
    `<div style="font-size:15px;margin:4px 0"><b>${r.n_hits}</b> / ${r.n_traces} traces match <b>/${r.pattern}/</b></div>
     <div class="dim" style="margin-bottom:12px"><span class="speed">scanned in ${r.elapsed_ms} ms — no model call</span>${top ? ` · ${top.model.split("-").pop()} most affected (${(top.rate * 100).toFixed(0)}%)` : ""}</div>
     <div class="eyebrow">which models</div>
     ${r.by_model.map((m) => `<div class="rrow"><span class="rlab">${m.model.split("-").slice(-2).join("-")}</span><span class="barbg"><span class="fill" style="width:${(m.rate * 160).toFixed(0)}px"></span></span><span>${(m.rate * 100).toFixed(0)}%</span></div>`).join("")}
     <div class="eyebrow">action mix · matched vs. all</div>
     <div class="rrow"><span class="rlab">matched</span>${r.n_hits ? mixbar(r.mix_hits, col) : '<span class="dim">no matches</span>'}</div>
     <div class="rrow"><span class="rlab">all traces</span>${mixbar(r.mix_all, col)}</div>
     <div class="eyebrow">matching traces</div>
     ${r.hits.map((h) => `<div class="qhit"><span style="min-width:48px">${h.model.split("-").pop()}</span><span class="dim">${h.atoms.length} steps</span>${spine(h.atoms, col)}</div>`).join("")}
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

loadDatasets().then(() => run(SAMPLES[0].pat));
