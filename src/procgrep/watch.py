"""Intent: procgrep watch, the live-attach companion to the recorded replay in
the explorer. Tail a trajectory source, push each new action atom over SSE to a
local page, and watch the trail build in real time as an agent works. Run it in
the terminal next to a rollout, open the printed localhost URL.

Design decisions (benefit / price):

1. Local, self-contained http.server plus SSE; stdlib only, no external deps and
   no write path on the hosted Space.
   Benefit: run it beside your agent, open localhost, watch; nothing to deploy
   and no public write surface to secure.
   Price: a single-process, single-stream demo, not a hosted multi-user service.

2. Two sources: --demo replays a built-in atom sequence at a cadence, and
   --tail FILE follows a file where each appended line is one action atom.
   Benefit: --demo proves the pipeline with no agent running; --tail is the real
   hook a rollout (or a thin wrapper) writes to as it acts.
   Price: turning a specific agent's native trajectory into atoms is the
   wrapper's job (it calls procgrep.canonicalize); this tool consumes atoms.

3. The page reuses the explorer's barcode plus terminal-line vocabulary inline.
   Benefit: one visual language across recorded replay and live-attach.
   Price: a small amount of render code is duplicated here, kept minimal.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# A realistic built-in rollout for --demo: search, read, edit, test, recover, submit.
DEMO_ATOMS = [
    "think",
    "search_repo",
    "think",
    "read_file",
    "think",
    "read_file",
    "edit",
    "think",
    "run_test",
    "error",
    "think",
    "edit",
    "think",
    "run_test",
    "edit",
    "think",
    "run_test",
    "think",
    "submit",
]

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>procgrep watch</title>
<style>
:root{--paper:#F7F5F2;--ink:#14110E;--rule:#d9d4cc;--copper:#CB4D20;--olive:#585E53;--mono:ui-monospace,Menlo,monospace}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--mono);font-size:13px;line-height:1.5}
.wrap{max-width:760px;margin:0 auto;padding:40px 28px}
h1{font-family:Georgia,serif;font-weight:400;font-size:24px;margin:0 0 4px}
.dim{color:var(--olive)}
.bc{display:inline-flex;gap:0;flex-wrap:wrap;margin:10px 0}
.bc i{width:5px;height:15px}
.thread{display:flex;flex-direction:column;gap:3px;margin-top:12px}
.tstep{display:flex;gap:8px;align-items:baseline}
.tstep .ti{width:22px;color:#b7b1a7;text-align:right;flex:none}
.tstep .dot{width:8px;height:8px;border-radius:2px;align-self:center;flex:none}
.fold{color:var(--olive);font-style:italic}
.cur{color:var(--copper);animation:blink 1s step-end infinite}
@keyframes blink{50%{opacity:0}}
.q{margin:16px 0;display:flex;gap:8px;align-items:center}
.q input{font-family:var(--mono);font-size:12px;padding:5px 8px;border:1px solid var(--rule);width:220px}
.fire{color:var(--copper)}
</style></head><body><div class="wrap">
<h1>procgrep watch</h1>
<div class="dim" id="status">connecting to the live stream…</div>
<div id="bc" class="bc"></div>
<div class="q"><span class="dim">live query</span> <input id="q" placeholder="(edit ){2,}"> <span id="fire" class="dim"></span></div>
<div class="thread" id="term"></div>
<script>
const COLOR={search_repo:"#585E53",read_file:"#5692E5",edit:"#CB4D20",create_file:"#3D7AD8",run_test:"#20A380",submit:"#14110E",think:"#b7b1a7",localize:"#8C1040",delete_file:"#A03D18",error:"#B4184F",other:"#d9d4cc"};
const PHRASE={search_repo:"grepped the repo",read_file:"opened a file",edit:"edited code",create_file:"created a file",delete_file:"deleted a file",run_test:"ran the tests",submit:"submitted the patch",localize:"localized the fault",error:"hit an error",think:"reasoning",other:"other"};
let atoms=[];
function draw(){
  document.getElementById("bc").innerHTML=atoms.map((a,i)=>{const noise=a==="think"||a==="other";return `<i title="${a}" style="background:${noise?"#e6e1d8":(COLOR[a]||"#ccc")}"></i>`;}).join("");
  const out=[];let i=0;
  while(i<atoms.length){const a=atoms[i];
    if(a==="think"||a==="other"){let k=0;while(i<atoms.length&&(atoms[i]==="think"||atoms[i]==="other")){k++;i++;}out.push(`<div class="tstep fold">- ${k} reasoning ${k>1?"steps":"step"}</div>`);}
    else{const now=i===atoms.length-1?' <span class="cur">|</span>':'';out.push(`<div class="tstep"><span class="ti">${i+1}</span><span class="dot" style="background:${COLOR[a]||"#ccc"}"></span>${PHRASE[a]||a}${now}</div>`);i++;}}
  const t=document.getElementById("term");t.innerHTML=out.join("");
  const pat=document.getElementById("q").value.trim();const el=document.getElementById("fire");
  if(!pat){el.textContent="";}else{let rx;try{rx=new RegExp(pat);}catch{el.textContent="…";return;}
    const sp=atoms.join(" ")+" ";const m=rx.exec(sp);el.innerHTML=m?`<span class="fire">● fired at step ${sp.slice(0,m.index).split(" ").length}</span>`:'<span class="dim">no match yet</span>';}
}
document.getElementById("q").addEventListener("input",draw);
const es=new EventSource("/events");
es.onmessage=(e)=>{const d=JSON.parse(e.data);if(d.atom){atoms.push(d.atom);document.getElementById("status").textContent=`${atoms.length} steps streamed`;draw();}if(d.done){document.getElementById("status").textContent=`done, ${atoms.length} steps`;es.close();}};
es.onerror=()=>{document.getElementById("status").textContent="stream closed";};
</script></div></body></html>"""


class _Bus:
    """Fan-out of atom events to the single connected SSE client."""

    def __init__(self) -> None:
        self.q: queue.Queue[dict[str, object]] = queue.Queue()

    def emit(self, event: dict[str, object]) -> None:
        self.q.put(event)


def _producer(bus: _Bus, args: argparse.Namespace) -> None:
    """Feed atoms into the bus from the chosen source."""
    if args.tail:
        # Follow a file; each appended line is one atom (the rollout, or a thin
        # wrapper calling procgrep.canonicalize, writes atoms as it acts).
        with open(args.tail) as fh:
            fh.seek(0, 2)  # start at end; only stream new lines
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                atom = line.strip()
                if atom:
                    bus.emit({"atom": atom})
    else:
        for atom in DEMO_ATOMS:
            bus.emit({"atom": atom})
            time.sleep(args.interval)
        bus.emit({"done": True})


def _make_handler(bus: _Bus, args: argparse.Namespace) -> type[BaseHTTPRequestHandler]:
    started = threading.Event()  # start the stream on first connect, so the page sees it paced

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:  # quiet
            return

        def do_GET(self) -> None:
            if self.path == "/events":
                if not started.is_set():
                    started.set()
                    threading.Thread(target=_producer, args=(bus, args), daemon=True).start()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while True:
                    event = bus.q.get()
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                    if event.get("done"):
                        break
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(PAGE.encode())

    return Handler


def serve(
    *,
    tail: str | None = None,
    demo: bool = False,
    port: int = 7870,
    interval: float = 0.6,
) -> int:
    """Serve the live watch page and stream atoms into it until interrupted.

    Exactly one source: ``tail`` follows a file where each appended line is
    one action atom; ``demo`` (or neither flag) replays the built-in rollout.
    """
    args = argparse.Namespace(tail=tail, demo=demo or not tail, port=port, interval=interval)
    bus = _Bus()
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(bus, args))
    source = f"tail {tail}" if tail else "demo"
    print(f"procgrep watch on http://127.0.0.1:{port}  (source: {source})")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0
