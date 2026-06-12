#!/usr/bin/env bash
# Regenerate the replay GIF used in the essay. Renders the demo trajectory at
# each step as a deterministic frame, then assembles them into a looping GIF
# that shows the trail building and the live query firing mid-run.
# Usage: bash scripts/capture_replay.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="$HOME/Library/Caches/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-mac-arm64/chrome-headless-shell"
FFMPEG="$(command -v ffmpeg || echo /opt/homebrew/bin/ffmpeg)"
FR=/tmp/procgrep_replay_frames
rm -rf "$FR"; mkdir -p "$FR"

cat > "$FR/frame.html" <<'EOF'
<!doctype html><html><head><meta charset="utf-8"><style>
:root{--paper:#F7F5F2;--ink:#14110E;--rule:#d9d4cc;--copper:#CB4D20;--olive:#585E53;--mono:ui-monospace,Menlo,monospace}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--mono);font-size:13px;line-height:1.5}
.panel{max-width:720px;margin:0 auto;padding:24px 28px}
h1{font-family:Georgia,serif;font-weight:400;font-size:21px;margin:0 0 2px}
.dim{color:var(--olive)}
.bc{display:inline-flex;gap:0;margin:8px 0}
.bc i{width:6px;height:15px}
.q{margin:10px 0;display:flex;gap:8px;align-items:center;font-size:12px}
.q input{font-family:var(--mono);font-size:12px;padding:4px 8px;border:1px solid var(--rule);width:230px;background:#fff}
.fire{color:var(--copper)}
.term{min-height:300px;display:flex;flex-direction:column;gap:3px;margin-top:8px}
.tstep{display:flex;gap:8px;align-items:baseline}
.tstep .ti{width:22px;color:#b7b1a7;text-align:right;flex:none}
.tstep .dot{width:8px;height:8px;border-radius:2px;align-self:center;flex:none}
.fold{color:var(--olive);font-style:italic}
.cur{color:var(--copper)}
</style></head><body><div class="panel">
<h1>Replaying one trajectory</h1>
<div class="dim" id="status"></div>
<div id="bc" class="bc"></div>
<div class="q"><span class="dim">live query</span> <input id="q" value="edit (?:think )?run_test" readonly> <span id="fire" class="dim"></span></div>
<div class="term" id="term"></div>
</div>
<script>
const ATOMS=["think","search_repo","think","read_file","think","read_file","edit","think","run_test","error","think","edit","think","run_test","edit","think","run_test","think","submit"];
const COLOR={search_repo:"#585E53",read_file:"#5692E5",edit:"#CB4D20",run_test:"#20A380",submit:"#14110E",think:"#b7b1a7",error:"#B4184F",other:"#d9d4cc"};
const PHRASE={search_repo:"grepped the repo",read_file:"opened a file",edit:"edited code",run_test:"ran the tests",submit:"submitted the patch",error:"hit an error",think:"reasoning",other:"other"};
const n=Math.min(+(new URLSearchParams(location.search).get("n")||ATOMS.length),ATOMS.length);
document.getElementById("bc").innerHTML=ATOMS.map((a,i)=>{const noise=a==="think"||a==="other";return `<i style="background:${noise?"#e6e1d8":(COLOR[a]||"#ccc")};opacity:${i<n?1:0.16}"></i>`;}).join("");
document.getElementById("status").textContent=`step ${n} of ${ATOMS.length}`;
const out=[];let i=0;
while(i<n){const a=ATOMS[i];
  if(a==="think"||a==="other"){let k=0;while(i<n&&(ATOMS[i]==="think"||ATOMS[i]==="other")){k++;i++;}out.push(`<div class="tstep fold">- ${k} reasoning ${k>1?"steps":"step"}</div>`);}
  else{const now=i===n-1?' <span class="cur">|</span>':"";out.push(`<div class="tstep"><span class="ti">${i+1}</span><span class="dot" style="background:${COLOR[a]||"#ccc"}"></span>${PHRASE[a]||a}${now}</div>`);i++;}}
document.getElementById("term").innerHTML=out.join("");
const sp=ATOMS.slice(0,n).join(" ")+" ";const m=new RegExp("edit (?:think )?run_test").exec(sp);const el=document.getElementById("fire");
el.innerHTML=m?`<span class="fire">● fired at step ${sp.slice(0,m.index).split(" ").length}</span>`:'<span class="dim">no match yet</span>';
</script></body></html>
EOF

for n in $(seq 1 19); do
  printf -v idx "%02d" "$n"
  "$BIN" --headless --disable-gpu --no-sandbox --hide-scrollbars --force-device-scale-factor=2 \
    --window-size=780,520 --virtual-time-budget=1500 \
    --screenshot="$FR/f$idx.png" "file://$FR/frame.html?n=$n" >/dev/null 2>&1
done
# hold the final frame
for h in 20 21 22 23; do cp "$FR/f19.png" "$FR/f$h.png"; done

"$FFMPEG" -y -framerate 1.8 -i "$FR/f%02d.png" \
  -vf "scale=760:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=dither=none" \
  -loop 0 docs/figures/replay.gif >/dev/null 2>&1
echo "wrote docs/figures/replay.gif ($(du -h docs/figures/replay.gif | cut -f1))"
