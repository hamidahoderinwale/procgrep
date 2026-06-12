"""Build the interactive web-article version of the procgrep paper.

A single self-contained HTML page (Distill x Nota aesthetic): the paper reads
end-to-end, figures inline, a live query-playground showcase near the top, and
(in later phases) interactive figures + reading affordances.

Pipeline: pandoc renders src/main.tex -> body fragment (citeproc + MathML); we
extract the abstract, build a TOC, fix figure paths, and wrap it in a styled
shell. Zero runtime dependencies; hostable on GitHub Pages.

    python docs/paper/build_paper.py            # -> docs/paper/index.html
"""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "main.tex"
BIB = ROOT / "src" / "main.bib"
OUT = ROOT.parent / "index.html"  # docs/index.html — the procgrep Pages root

TITLE = "Agent trajectories as programs"
AUTHOR = "Hamidah Oderinwale"
SUBTITLE = "Fingerprinting and programming coding-agent behavior."


def run_pandoc() -> str:
    """Render the .tex body to an HTML fragment (no standalone wrapper)."""
    cmd = [
        "pandoc",
        str(SRC),
        "-f",
        "latex",
        "-t",
        "html5",
        "--katex",
        "--citeproc",
        f"--bibliography={BIB}",
        "--wrap=none",
        "--section-divs",
        "-M",
        "link-citations=true",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def delatex(s: str) -> str:
    """Light LaTeX -> HTML for the abstract (pandoc drops it as metadata)."""
    s = re.sub(r"\\footnote\{.*?\}", "", s, flags=re.S)
    s = re.sub(r"\\texttt\{(.*?)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\emph\{(.*?)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\textbf\{(.*?)\}", r"<strong>\1</strong>", s)
    s = re.sub(r"\\url\{(.*?)\}", r'<a href="\1">\1</a>', s)
    s = s.replace(r"\%", "%").replace(r"\&", "&amp;").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+\{(.*?)\}", r"\1", s)  # strip any remaining \cmd{...}
    s = s.replace("{", "").replace("}", "")  # drop residual stray braces
    return " ".join(s.split())


def extract_abstract() -> str:
    tex = SRC.read_text()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.S)
    return delatex(m.group(1)) if m else ""


def build_toc(body: str) -> str:
    """Numbered TOC from top-level sections (section-divs give us ids)."""
    secs = re.findall(r'<section id="([^"]+)"[^>]*>\s*<h1[^>]*>(.*?)</h1>', body, flags=re.S)
    items = []
    n = 0
    for sid, title in secs:
        title = re.sub(r"<[^>]+>", "", title).strip()
        n += 1
        items.append(
            f'<li><a href="#{sid}"><span class="tocn">{n:02d}</span>'
            f'<span class="toct">{html.escape(title)}</span></a></li>'
        )
    # References is a div, not a <section> — add it explicitly so nothing is hidden
    items.append(
        '<li><a href="#references"><span class="tocn">★</span>'
        '<span class="toct">References</span></a></li>'
    )
    return (
        "<nav class='toc'><div class='toclab'>Contents</div><ol>" + "".join(items) + "</ol></nav>"
    )


def fix_figures(body: str) -> str:
    """Wrap pandoc <figure>/<img> so they sit as full-column article figures."""
    # pandoc emits figures under figures/<name>.png already; ensure lazy load.
    body = body.replace("<img ", '<img loading="lazy" ')
    return body


SHOWCASE = """
<aside class="showcase" id="showcase">
  <div class="sc-head"><span class="sc-eyebrow">procgrep demo · try it live</span>
    <span><a class="sc-open" href="https://midah-procgrep-explorer.hf.space" target="_blank" rel="noopener">query ↗</a> &nbsp;·&nbsp; <a class="sc-open" href="https://midah-procgrep-explorer.hf.space/#compare" target="_blank" rel="noopener">compare agents ↗</a></span></div>
  <div class="sc-frame">
    <iframe id="scframe" src="explorer/index.html#query" title="ProcGrep query playground" loading="lazy"></iframe>
  </div>
  <p class="caption">Embedded above: ask a structural question over a whole dataset, answered in milliseconds with no model call. The live tool also sets two agents side by side, and compares two evals or resolved versus unresolved runs, with a diff strip that names the procedures separating them.</p>
</aside>
<figure class="exfig" id="compare-fig">
  <img src="figures/explorer_compare.png" alt="The compare view of the live explorer, two agents side by side" loading="lazy">
  <figcaption class="caption">The compare view. Two groups side by side, each row one trajectory drawn as a barcode of canonical actions, reasoning pale and actions vivid, with a diff strip that names the procedures most separating them and the gaps in length and reasoning. Toggle by agent, by eval, or by resolved versus unresolved runs. <a href="https://midah-procgrep-explorer.hf.space/#compare" target="_blank" rel="noopener">Open it live ↗</a></figcaption>
</figure>
"""

JSD_JS = r"""
(function(){
  const el=document.getElementById('jsd-data'); if(!el)return;
  const JSD=JSON.parse(el.textContent);
  const order=["Claude-3","Claude-3.5","Claude-3.7-thinking","Claude-4","GPT-4","GPT-4o","Agentless+Claude-3.5","DARS+R1","Moatless+V3"];
  const A=JSD.agents,M=JSD.matrix,ME=JSD.meta,S=JSD.axis_stats;
  const oi=order.map(a=>A.indexOf(a)).filter(i=>i>=0), N=oi.length, lab=oi.map(i=>A[i]);
  const mount=document.getElementById('fig-jsd'); if(!mount)return;
  const col=v=>{const t=Math.max(0,Math.min(1,v)),a=[230,240,236],b=[13,92,70];
    return 'rgb('+a.map((x,k)=>Math.round(x+(b[k]-x)*t)).join(',')+')';};
  const CELL=44,L=158,B=118,W=L+N*CELL+18,H=N*CELL+B;
  let axis=null;
  mount.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block;margin:0 auto"></svg>`+
    `<div class="jtip" id="jsd-tip"><span class="dim">Pairwise procedural divergence across nine agents. Hover any cell; toggle an axis below to see what predicts similarity.</span></div>`+
    `<div class="jctl"><span class="jlab">highlight axis:</span><span id="jsd-btns"></span></div>`;
  const svg=mount.querySelector('svg'),tip=document.getElementById('jsd-tip'),btns=document.getElementById('jsd-btns');
  function draw(){
    let s='';
    for(let r=0;r<N;r++)for(let c=0;c<N;c++){
      const ri=oi[r],ci=oi[c],v=M[ri][ci],x=L+c*CELL,y=8+r*CELL;
      const same=axis&&r!==c&&ME[ri][axis]===ME[ci][axis];
      s+=`<rect class="jc" x="${x}" y="${y}" width="${CELL-1.5}" height="${CELL-1.5}" fill="${col(v)}" `+
         `stroke="${same?'#CB4D20':'#fff'}" stroke-width="${same?2.5:1}" data-r="${ri}" data-c="${ci}"/>`;
      if(r===c)s+=`<text x="${x+CELL/2}" y="${y+CELL/2+3}" text-anchor="middle" font-size="9" fill="#aaa">0</text>`;
    }
    for(let r=0;r<N;r++)s+=`<text x="${L-8}" y="${8+r*CELL+CELL/2+3}" text-anchor="end" font-size="11">${lab[r]}</text>`;
    for(let c=0;c<N;c++){const cx=L+c*CELL+CELL/2,cy=8+N*CELL+10;
      s+=`<text x="${cx}" y="${cy}" text-anchor="end" font-size="11" transform="rotate(-45 ${cx} ${cy})">${lab[c]}</text>`;}
    svg.innerHTML=s;
    btns.innerHTML=['scaffold','era','family'].map(k=>`<button class="jbtn ${axis===k?'on':''}" data-k="${k}">${k}</button>`).join(' ')+
      (axis?`<span class="jstat">same <b>${axis}</b> ${S[axis].within} · different ${S[axis].across} <span class="dim">(${S[axis].scope})</span></span>`:'');
  }
  btns.addEventListener('click',e=>{const k=e.target.dataset.k;if(k){axis=axis===k?null:k;draw();}});
  svg.addEventListener('mousemove',e=>{const t=e.target;if(t.tagName!=='rect'){return;}
    const ri=+t.dataset.r,ci=+t.dataset.c,v=M[ri][ci],a=ME[ri],b=ME[ci];
    if(ri===ci){tip.innerHTML='<span class="dim">an agent vs itself — JSD 0 by definition</span>';return;}
    tip.innerHTML=`<b>${a.agent}</b> vs <b>${b.agent}</b> &nbsp; JSD <b>${v.toFixed(2)}</b><br>`+
      ['family','era','scaffold'].map(k=>`${k}: `+(a[k]===b[k]?`<span style="color:#20A380">same</span>`:`<span style="color:#CB4D20">differ</span>`)).join(' &nbsp;·&nbsp; ');});
  draw();
})();
"""

FT_JS = r"""
(function(){
  const el=document.getElementById('ft-data'); if(!el)return;
  const D=JSON.parse(el.textContent).agents;
  const mount=document.getElementById('fig-ft'); if(!mount)return;
  const FCOL={Claude:'#CB4D20',GPT:'#5692E5',Other:'#585E53'};
  const W=560,RH=34,PADL=160,PADR=20,PADT=18,PADB=46,H=PADT+D.length*RH+PADB;
  const x=v=>PADL+v*(W-PADL-PADR);
  mount.innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;display:block;margin:0 auto"></svg>`+
    `<div class="jtip" id="ft-tip"><span class="dim">Forward = the agent does what it said (says→does). Reverse = it said what it did (did→says). Hover a marker.</span></div>`+
    `<div class="ftleg"><span><svg width=14 height=14><circle cx=7 cy=7 r=5 fill="#585E53"/></svg> forward</span>`+
    `<span><svg width=14 height=14><polygon points="7,2 12,12 2,12" fill="#585E53"/></svg> reverse</span>`+
    `<span><i style="background:#CB4D20"></i>Claude</span><span><i style="background:#5692E5"></i>GPT</span></div>`;
  const svg=mount.querySelector('svg'),tip=document.getElementById('ft-tip');
  let s='';
  [0,0.5,1].forEach(t=>{s+=`<line x1="${x(t)}" y1="${PADT}" x2="${x(t)}" y2="${H-PADB+4}" stroke="#e6e1d8"/>`+
    `<text x="${x(t)}" y="${H-PADB+18}" text-anchor="middle" font-size="11" fill="#585E53">${t}</text>`;});
  s+=`<text x="${(PADL+W-PADR)/2}" y="${H-6}" text-anchor="middle" font-size="11" fill="#585E53">action-account coverage</text>`;
  D.forEach((d,i)=>{const cy=PADT+i*RH+RH/2,c=FCOL[d.family];
    s+=`<text x="${PADL-10}" y="${cy+3}" text-anchor="end" font-size="11">${d.agent}</text>`;
    s+=`<line x1="${x(d.fwd_iqr[0])}" y1="${cy-6}" x2="${x(d.fwd_iqr[1])}" y2="${cy-6}" stroke="${c}" stroke-opacity=".4"/>`;
    s+=`<circle class="fc" cx="${x(d.fwd)}" cy="${cy-6}" r="5.5" fill="${c}" data-i="${i}"/>`;
    s+=`<line x1="${x(d.rev_iqr[0])}" y1="${cy+7}" x2="${x(d.rev_iqr[1])}" y2="${cy+7}" stroke="${c}" stroke-opacity=".4"/>`;
    const tx=x(d.rev);s+=`<polygon class="fc" points="${tx},${cy+2} ${tx+5.5},${cy+12} ${tx-5.5},${cy+12}" fill="${c}" data-i="${i}"/>`;
  });
  svg.innerHTML=s;
  svg.addEventListener('mousemove',e=>{const t=e.target;if(!t.classList||!t.classList.contains('fc'))return;
    const d=D[+t.dataset.i];
    tip.innerHTML=`<b>${d.agent}</b> &nbsp; forward <b>${d.fwd.toFixed(2)}</b> <span class="dim">[${d.fwd_iqr[0]}–${d.fwd_iqr[1]}]</span> &nbsp;·&nbsp; reverse <b>${d.rev.toFixed(2)}</b> <span class="dim">[${d.rev_iqr[0]}–${d.rev_iqr[1]}]</span>`;});
})();
"""

PIPE_JS = r"""
(function(){
  const mount=document.getElementById('fig-pipe'); if(!mount)return;
  const VC={localize:'#8C1040',read:'#5692E5',edit:'#CB4D20',test:'#20A380',submit:'#14110E'};
  const SC={A:'#5692E5',B:'#CB4D20',C:'#A98D5A'};
  const trace=[
    {op:'localize_file("auth.py")',verb:'localize',sub:'A'},
    {op:'read_file("auth.py")',verb:'read',sub:'A'},
    {op:'edit_file("auth.py", p1)',verb:'edit',sub:'B'},
    {op:'run_tests("test_auth")',verb:'test',sub:'B'},
    {op:'edit_file("auth.py", p2)',verb:'edit',sub:'B'},
    {op:'run_tests("test_auth")',verb:'test',sub:'C'},
    {op:'submit()',verb:'submit',sub:'C'}];
  const subs={A:['localize','read'],B:['edit','test','edit'],C:['test','submit']};
  const stages=[
    {t:'Raw tool calls',d:'The agent emits a stream of tool invocations — verbose, parameterized, every one different.'},
    {t:'Canonical verbs',d:'Each call is normalized to a canonical action; surface detail dropped, the verb kept.'},
    {t:'Merge co-occurring',d:'Frequently-adjacent verbs are merged into recurring subroutines — exactly as BPE merges tokens.'},
    {t:'Induced vocabulary',d:'The merges define an alphabet of procedures, learned bottom-up from the corpus.'},
    {t:'Encoded trace',d:'The trajectory is now a short string over that alphabet — a program we can compare and query.'}];
  let st=0;
  const chip=(l,c)=>`<span class="pchip" style="background:${c}">${l}</span>`;
  function render(){
    let b='';
    if(st===0) b=trace.map(r=>`<div class="prow"><code>${r.op}</code></div>`).join('');
    else if(st===1) b='<div class="pspine">'+trace.map(r=>chip(r.verb,VC[r.verb])).join('')+'</div>';
    else if(st===2) b='<div class="pspine">'+trace.map(r=>`<span class="pwrap" style="border-color:${SC[r.sub]}">`+chip(r.verb,VC[r.verb])+'</span>').join('')+'</div><div class="pdim">brackets = the subroutine each verb merges into</div>';
    else if(st===3) b='<div class="pvocab">'+Object.entries(subs).map(([k,v])=>`<div class="pvrow">${chip(k,SC[k])} = <span class="pdim" style="margin:0">{ ${v.join(', ')} }</span></div>`).join('')+'</div>';
    else b='<div class="pspine big">'+['A','B','C'].map(k=>chip(k,SC[k])).join('')+'</div><div class="pdim">localize→read · edit→test→edit · test→submit</div>';
    mount.querySelector('#pipe-body').innerHTML=b;
    mount.querySelector('#pipe-desc').textContent=stages[st].d;
    mount.querySelectorAll('.ptab').forEach((x,i)=>x.classList.toggle('on',i===st));
  }
  mount.innerHTML=`<div class="ptabs">${stages.map((s,i)=>`<button class="ptab" data-i="${i}">${i+1}. ${s.t}</button>`).join('')}</div>`+
    `<div id="pipe-body" class="pbody"></div><div id="pipe-desc" class="pdim" style="margin-top:10px"></div>`+
    `<div class="pnav"><button id="pplay">❚❚ pause</button><button id="pprev">‹ prev</button><button id="pnext">next ›</button></div>`;
  // GIF-like autoplay: loop the stages; any manual control pauses it.
  let timer=null, playing=false;
  function play(){playing=true;mount.querySelector('#pplay').textContent='❚❚ pause';
    timer=setInterval(()=>{st=(st+1)%stages.length;render();},2000);}
  function pause(){playing=false;mount.querySelector('#pplay').textContent='▶ play';
    if(timer){clearInterval(timer);timer=null;}}
  mount.querySelector('.ptabs').addEventListener('click',e=>{const i=e.target.dataset.i;if(i!=null){pause();st=+i;render();}});
  mount.querySelector('#pprev').onclick=()=>{pause();st=(st+stages.length-1)%stages.length;render();};
  mount.querySelector('#pnext').onclick=()=>{pause();st=(st+1)%stages.length;render();};
  mount.querySelector('#pplay').onclick=()=>{playing?pause():play();};
  // start playing only when scrolled into view, so it's not spinning off-screen
  render();
  if('IntersectionObserver' in window){
    const io=new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting&&!playing&&!timer)play();else if(!e.isIntersecting&&playing)pause();});},{threshold:0.3});
    io.observe(mount);
  } else play();
})();
"""

STYLE = """
:root{--paper:#F7F5F2;--ink:#14110E;--rule:#d9d4cc;--copper:#CB4D20;--blue:#5692E5;
--teal:#20A380;--olive:#585E53;--gray:#b7b1a7;
--serif:'Charter','Iowan Old Style','Palatino Linotype',Palatino,'Spectral',Georgia,serif;
--mono:ui-monospace,"SFMono-Regular",Menlo,monospace}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--serif);
font-size:18px;line-height:1.65;-webkit-font-smoothing:antialiased}
.topbar{position:sticky;top:0;z-index:20;background:rgba(247,245,242,.92);backdrop-filter:blur(6px);
font-family:var(--mono);font-size:13px}
.progress{position:fixed;top:0;left:0;height:2px;width:0;background:var(--copper);z-index:40;transition:width .1s linear}
.topbar .in{max-width:1080px;margin:0 auto;padding:11px 32px;display:flex;justify-content:space-between;align-items:baseline}
.topbar .mark{font-weight:600}.topbar a{color:var(--olive);text-decoration:none;margin-left:18px}
.topbar a:hover{color:var(--ink)}
:root{--col:1020px;--gutter:300px}
.hero{max-width:var(--col);margin:0 auto;padding:64px var(--gutter) 8px 32px}
.hero h1{font-size:42px;line-height:1.12;font-weight:600;margin:0 0 14px;letter-spacing:-.01em}
.hero .sub{font-size:20px;color:var(--olive);margin:0 0 22px;line-height:1.4}
.hero .byline{font-family:var(--mono);font-size:13px;color:var(--olive);padding:4px 0;display:flex;gap:18px;flex-wrap:wrap}
.hero .byline a{color:var(--copper);text-decoration:none}
.abstract{max-width:var(--col);margin:30px auto 0;padding:0 var(--gutter) 0 32px}
.abstract .lab{font-family:var(--mono);text-transform:uppercase;letter-spacing:.12em;font-size:11px;color:var(--olive);margin-bottom:8px}
.abstract p{font-size:15px;line-height:1.62;margin:0;color:#2a2520}
/* showcase */
.showcase{max-width:1020px;margin:40px auto;padding:22px;border:1px solid var(--ink);
background:#fff;border-radius:4px}
.sc-head{display:flex;justify-content:space-between;align-items:baseline}
.sc-eyebrow{font-family:var(--mono);text-transform:uppercase;letter-spacing:.12em;font-size:11px;color:var(--copper)}
.sc-open{font-family:var(--mono);font-size:12px;color:var(--copper);text-decoration:none}
.sc-open:hover{text-decoration:underline}
.exfig{max-width:1020px;margin:26px auto;padding:0}
.exfig img{width:100%;height:auto;display:block;border:1px solid var(--rule)}
.exfig figcaption{margin-top:8px}
.sc-title{font-size:26px;font-weight:600;margin:6px 0 6px}
.sc-sub{font-size:16px;color:var(--olive);max-width:64ch;margin:0 0 14px}
.sc-frame{border:1px solid var(--rule);border-radius:3px;overflow:hidden;background:var(--paper)}
.sc-frame iframe{width:100%;height:560px;border:0;display:block}
.sc-note{font-family:var(--mono);font-size:12px;color:var(--olive);margin-top:8px}
.sc-note a{color:var(--copper)}
/* TOC */
.toc{max-width:var(--col);margin:30px auto 12px;padding:0 var(--gutter) 0 32px}
.toc .toclab{font-family:var(--mono);text-transform:uppercase;letter-spacing:.12em;font-size:11px;color:var(--olive);margin-bottom:12px}
.toc ol{list-style:none;margin:0;padding:0;columns:2;column-gap:44px}
.toc li{margin:0 0 8px;break-inside:avoid}
.toc a{color:var(--ink);text-decoration:none;display:flex;gap:11px;align-items:baseline;font-size:15px;line-height:1.35}
.toc a:hover .toct{color:var(--copper)}
.toc .tocn{font-family:var(--mono);font-size:12px;color:var(--copper);min-width:18px;flex:none}
.toc .toct{transition:color .12s}
/* article body */
article{max-width:var(--col);margin:0 auto;padding:10px var(--gutter) 90px 32px;position:relative}
article h1{font-size:28px;font-weight:600;margin:60px 0 12px;letter-spacing:-.01em}
article h2{font-size:21px;font-weight:600;margin:38px 0 10px}
article h3{font-size:17px;font-family:var(--mono);font-weight:600;margin:26px 0 8px;color:var(--olive)}
article p{margin:0 0 18px}
article a{color:var(--copper)}
article code{font-family:var(--mono);font-size:.86em;background:#efeae2;padding:1px 5px;border-radius:2px}
article figure{margin:30px 0;text-align:center}
article figure img{max-width:100%;height:auto;border:1px solid var(--rule);background:#fff;border-radius:3px}
article figure figcaption,article .caption{font-family:var(--mono);font-size:13px;color:var(--olive);
line-height:1.5;margin-top:10px;text-align:left;max-width:62ch;margin-left:auto;margin-right:auto}
/* tables */
article table{width:100%;border-collapse:collapse;margin:24px 0;font-family:var(--mono);font-size:13px}
article th{text-align:left;font-weight:600;color:var(--olive);border-bottom:1px solid var(--ink);padding:6px 8px}
article td{border-bottom:1px solid var(--rule);padding:6px 8px;vertical-align:top}
/* the procgrep example query box */
article .tcolorbox{background:#fff;border:1px solid var(--copper);border-left:3px solid var(--copper);
border-radius:3px;padding:12px 16px;margin:22px 0;font-family:var(--mono);font-size:14px;color:var(--ink)}
/* citations + references */
.citation{font-family:var(--mono);font-size:.8em;color:var(--copper)}
#refs{font-size:14px;line-height:1.5}
#refs .csl-entry{margin:0 0 10px;padding-left:1.4em;text-indent:-1.4em}
math{font-family:var(--serif)}
/* interactive figures */
.interactive{margin:30px 0;padding:16px;border:1px solid var(--rule);border-radius:3px;background:#fff}
.interactive svg text{font-family:var(--mono)}
.jc{cursor:crosshair}
.jtip{font-family:var(--mono);font-size:13px;color:var(--ink);min-height:38px;margin:2px 0 6px;line-height:1.5}
.jtip .dim{color:var(--olive)}
.jctl{font-family:var(--mono);font-size:13px;margin-top:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.jlab{color:var(--olive);text-transform:uppercase;letter-spacing:.1em;font-size:11px}
.jbtn{font-family:var(--mono);font-size:12px;padding:3px 11px;border:1px solid var(--rule);background:var(--paper);cursor:pointer;border-radius:2px;color:var(--ink)}
.jbtn:hover{border-color:var(--ink)}
.jbtn.on{background:var(--copper);color:#fff;border-color:var(--copper)}
.jstat{margin-left:4px}.jstat .dim{color:var(--olive)}
.ftleg{font-family:var(--mono);font-size:12px;color:var(--olive);display:flex;gap:16px;flex-wrap:wrap;margin-top:8px}
.ftleg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
.ftleg svg{vertical-align:middle}.fc{cursor:crosshair}
.ptabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.ptab{font-family:var(--mono);font-size:12px;padding:4px 10px;border:1px solid var(--rule);background:var(--paper);cursor:pointer;border-radius:2px;color:var(--olive)}
.ptab.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.pbody{min-height:160px;display:flex;flex-direction:column;justify-content:center;align-items:center;padding:14px;background:var(--paper);border-radius:3px}
.prow{font-family:var(--mono);font-size:13px;margin:2px 0}.prow code{background:none;padding:0}
.pspine{display:flex;flex-wrap:wrap;gap:6px;justify-content:center}
.pspine.big{gap:14px}.pspine.big .pchip{font-size:18px;padding:8px 18px}
.pchip{font-family:var(--mono);font-size:12px;color:#fff;padding:3px 9px;border-radius:3px}
.pwrap{display:inline-block;border:2px solid;border-radius:5px;padding:2px}
.pvocab{font-family:var(--mono);font-size:14px}.pvrow{margin:7px 0}
.pdim{font-family:var(--mono);font-size:12px;color:var(--olive);text-align:center}
.pnav{display:flex;gap:8px;justify-content:center;margin-top:12px}
.pnav button{font-family:var(--mono);font-size:12px;padding:4px 14px;border:1px solid var(--rule);background:#fff;cursor:pointer;border-radius:2px}
.pnav button:hover{border-color:var(--ink)}
/* margin citations (Tufte sidenotes) — float into the right whitespace on wide screens */
.sidenote{float:right;clear:right;width:250px;margin:.2em -288px 1.2em 0;
  font-family:var(--mono);font-size:10.5px;line-height:1.5;color:var(--olive)}
.sidenote .sn-ref{display:block;margin-bottom:9px;padding-left:10px;border-left:1px solid var(--rule)}
.sidenote .sn-ref:last-child{margin-bottom:0}
.sidenote a{color:var(--copper);word-break:break-word}
@media(max-width:1100px){
  .hero,.abstract,.toc,article{padding-right:32px}
  /* no room for a margin — drop the sidenotes; the in-text citation is still a hyperlink to References */
  .sidenote{display:none}}
/* grounded tooltips */
.term{border-bottom:1px dotted var(--olive);cursor:help;position:relative}
.term .tip{position:absolute;left:0;top:1.6em;z-index:30;width:272px;background:var(--ink);color:var(--paper);
  font-family:var(--mono);font-size:11px;line-height:1.5;padding:10px 12px;border-radius:5px;
  box-shadow:0 6px 18px rgba(20,17,14,.2);opacity:0;visibility:hidden;transition:opacity .12s}
.term:hover .tip,.term:focus .tip{opacity:1;visibility:visible}
.term .tip-def{display:block}
.term .tip-src{display:block;margin-top:7px;color:#b7b1a7}
.term .tip-src a{color:#e9b8a6}
.foot{max-width:var(--col);margin:40px auto 0;padding:24px var(--gutter) 70px 32px;
font-family:var(--mono);font-size:12px;color:var(--olive)}
@media(max-width:760px){.toc{columns:1}.hero h1{font-size:32px}}
/* code cards — light, Cursor-blog style (NOT a dark terminal theme) */
.codecard{margin:24px 0;border:1px solid var(--rule);border-radius:7px;background:#fff;
  overflow:hidden;box-shadow:0 1px 2px rgba(20,17,14,.04)}
.codecard-head{display:flex;align-items:center;gap:8px;padding:8px 14px;
  background:#f3efe8;border-bottom:1px solid var(--rule);
  font-family:var(--mono);font-size:11.5px;letter-spacing:.02em}
.codecard-dot{width:9px;height:9px;border-radius:50%;background:var(--copper);flex:none;opacity:.7}
.codecard-name{color:var(--ink);font-weight:600}
.codecard-lang{margin-left:auto;color:var(--olive);text-transform:uppercase;letter-spacing:.12em;font-size:10px}
.codecard-body{margin:0;padding:14px 16px;background:#fff;overflow-x:auto;
  font-family:var(--mono);font-size:13px;line-height:1.6;color:var(--ink);
  white-space:pre;tab-size:2}
.codecard-body code{background:none;padding:0;font-size:inherit;border-radius:0}
/* syntax tinting */
.cc-key{color:var(--copper)}
.cc-com{color:var(--olive);font-style:italic}
.cc-str{color:var(--teal)}
.cc-num{color:var(--blue)}
.cc-punc{color:var(--gray)}
.cc-kw{color:var(--copper);font-weight:600}
.cc-val{color:var(--ink)}
.cc-ln{color:var(--gray);user-select:none;display:inline-block;width:2.1em;text-align:right;
  margin-right:1.1em;font-size:11px}
.algocard .codecard-head{background:#efece5}
"""


def inject_interactive(body: str) -> tuple[str, str]:
    """Swap the static JSD figure for an interactive mount; return embedded data."""
    jsd = (ROOT / "data" / "jsd.json").read_text()
    ft = (ROOT / "data" / "followthrough.json").read_text()
    body = re.sub(
        r"<img[^>]*fig_jsd_matrix_full_canonical\.png[^>]*>",
        '<div id="fig-jsd" class="interactive"></div>',
        body,
    )
    body = re.sub(
        r"<img[^>]*cot_alignment_6agents\.png[^>]*>",
        '<div id="fig-ft" class="interactive"></div>',
        body,
    )
    body = re.sub(
        r"<img[^>]*trace_pipeline\.png[^>]*>", '<div id="fig-pipe" class="interactive"></div>', body
    )
    # give the bibliography a heading so the TOC anchor (#references) resolves
    body = body.replace('<div id="refs"', '<h1 id="references">References</h1>\n<div id="refs"', 1)
    data_script = (
        f'<script id="jsd-data" type="application/json">{jsd}</script>'
        f'<script id="ft-data" type="application/json">{ft}</script>'
    )
    return body, data_script


def margin_citations(body: str) -> str:
    """Float each in-text citation's full reference (ground-truth from the bib)
    into the right margin as a Tufte sidenote. In-text stays a hyperlink."""
    refs: dict[str, str] = {}
    for m in re.finditer(r'<div id="ref-([^"]+)"[^>]*>(.*?)</div>', body, flags=re.S):
        refs[m.group(1)] = " ".join(m.group(2).split())

    def add_note(match: re.Match) -> str:
        span, keys = match.group(0), match.group(1).split()
        notes = [f'<span class="sn-ref">{refs[k]}</span>' for k in keys if k in refs]
        return span + ('<span class="sidenote">' + "".join(notes) + "</span>" if notes else "")

    return re.sub(
        r'<span class="citation" data-cites="([^"]+)"[^>]*>.*?</span>', add_note, body, flags=re.S
    )


# term -> (grounded definition, source label, optional bib ref-key to link the source)
GLOSSARY = [
    (
        r"GroupKFold",
        "K-fold cross-validation in which all samples from one group — here, one task instance — stay in the same fold, so no instance is ever split across train and test.",
        "scikit-learn",
        None,
    ),
    (
        r"Jensen[–-]Shannon [Dd]ivergence",
        "A symmetric, smoothed measure of how different two probability distributions are: 0 when identical, and 1 (base-2) when they share no support.",
        "defined in this paper, §An information theory of procedures",
        None,
    ),
    (
        r"\bJSD\b",
        "Jensen–Shannon divergence — a symmetric distance between two probability distributions, 0 (identical) to 1 (disjoint, base-2).",
        "defined in this paper, §An information theory of procedures",
        None,
    ),
    (
        r"Byte[- ]Pair Encoding",
        "Iteratively merge the most frequent adjacent pair into a new token, building a compact vocabulary bottom-up; here applied to action sequences rather than text.",
        "Sennrich et al. 2016",
        "sennrich2016neuralmachinetranslationrare",
    ),
    (
        r"V-measure",
        "The harmonic mean of homogeneity and completeness — an entropy-based score for how well a clustering matches ground-truth labels.",
        "Rosenberg & Hirschberg 2007",
        "rosenberg2007vmeasure",
    ),
    (
        r"PrefixSpan",
        "A classic sequential-pattern miner that grows frequent subsequences by projecting the database on each pattern's prefix.",
        "standard sequential-pattern mining",
        None,
    ),
    (
        r"\bscaffold(s)?\b",
        "The wrapper around a model that determines which tools it can use, how it acquires context, and how it reacts to feedback.",
        "this paper, §Background",
        None,
    ),
    (
        r"\bfingerprint(s)?\b",
        "An agent's characteristic procedural habits — recurring patterns of tool use and ordering distinctive enough to identify the producing agent.",
        "this paper, Abstract",
        None,
    ),
    (
        r"\bentropy\b",
        "Shannon entropy H: the uncertainty, or spread, of a probability distribution, measured in bits.",
        "this paper, §An information theory of procedures",
        None,
    ),
    (
        r"macro-F1",
        "The F1 score averaged equally over classes, so rare and common agents count the same.",
        "standard classification metric",
        None,
    ),
    (
        r"abstract syntax tree(s)?",
        "AST — the parsed tree structure of source code, capturing its syntactic and semantic form independent of formatting.",
        "standard program analysis",
        None,
    ),
    (
        r"program synthesis",
        "Automatically generating programs from a specification — input/output examples, types, or a natural-language description.",
        "Austin et al. 2021",
        "austin2021programsynthesislargelanguage",
    ),
]


def glossary(body: str) -> str:
    """Wrap the first occurrence of each glossary term with a grounded tooltip."""
    for pat, definition, src, ref in GLOSSARY:
        srchtml = f'<a href="#ref-{ref}">{src}</a>' if ref else src
        tip = (
            f'<span class="tip"><span class="tip-def">{definition}</span>'
            f'<span class="tip-src">— {srchtml}</span></span>'
        )

        def wrap(m: re.Match, _t=tip) -> str:
            return f'<span class="term" tabindex="0">{m.group(0)}{_t}</span>'

        # first occurrence only, not inside a tag/attribute
        body = re.sub(rf"(?<![\w>])(?:{pat})(?![\w<])", wrap, body, count=1)
    return body


# --- code cards -------------------------------------------------------------

# Corrected, hand-formatted pseudocode for the two vocabulary-discovery
# algorithms (pandoc flattens the LaTeX algorithm environments into run-on
# text, so we rebuild them from scratch). Indentation is significant and is
# preserved verbatim inside the <pre> body of a code card.
ALGO_BPE = """\
Input:  action sequences S = {s1, ..., sn};  target vocabulary size K
Output: vocabulary V;  tokenized sequences T

V ← { all atoms appearing in S }
T ← S
while |V| < K:
    count frequency of every adjacent token pair (a, b) in T
    (a*, b*) ← argmax over (a, b) of freq(a, b)
    if freq(a*, b*) below threshold:
        break
    t ← a* · b*
    V ← V ∪ { t }
    replace every adjacent occurrence of (a*, b*) in T with t
return V, T
"""

ALGO_PREFIXSPAN = """\
Input:  sequences S;  minimum support σ
Output: frequent sequential patterns P

P ← ∅
for each item b that is frequent in the σ-projected database S|α:
    α' ← α · b
    P  ← P ∪ { α' }
    S|α' ← { suffix of r after its first occurrence of b : r ∈ S|α }
    recurse on S|α' to extend α'
return P
"""

# keywords highlighted in the pseudocode cards
_ALGO_KW = re.compile(r"\b(Input|Output|while|for each|if|break|return|recurse on)\b")


def _algo_card(label: str, code: str) -> str:
    """Render corrected pseudocode as a monospace code card (whitespace kept)."""
    esc = html.escape(code.rstrip("\n"))
    esc = _ALGO_KW.sub(r'<span class="cc-kw">\1</span>', esc)
    return (
        f'<div class="codecard algocard">'
        f'<div class="codecard-head"><span class="codecard-dot"></span>'
        f'<span class="codecard-name">{html.escape(label)}</span>'
        f'<span class="codecard-lang">pseudocode</span></div>'
        f'<pre class="codecard-body"><code>{esc}</code></pre></div>'
    )


_TINT_COMMENT = re.compile(r"(#.*)$")
_TINT_YAML_KEY = re.compile(r"^(\s*(?:- )?)([A-Za-z_][\w]*)(:)")
_TINT_NUM = re.compile(r"(?<![\w.])(\d+\.\d+|\d+)(?![\w.])")


def _tint_yaml_line(line: str) -> str:
    """Light YAML syntax tinting for one already-escaped line."""
    m = _TINT_COMMENT.search(line)
    comment = ""
    if m:
        comment = f'<span class="cc-com">{m.group(1)}</span>'
        line = line[: m.start()]
    line = _TINT_YAML_KEY.sub(
        lambda k: (
            f'{k.group(1)}<span class="cc-key">{k.group(2)}</span>'
            f'<span class="cc-punc">{k.group(3)}</span>'
        ),
        line,
    )
    line = _TINT_NUM.sub(r'<span class="cc-num">\1</span>', line)
    return line + comment


def _tint_json_line(line: str) -> str:
    """Light JSON tinting: quoted keys/strings and numbers."""
    line = re.sub(
        r"(&quot;[^&]*?&quot;)(\s*:)",
        r'<span class="cc-key">\1</span><span class="cc-punc">\2</span>',
        line,
    )
    line = re.sub(
        r"(?<![>\w])(&quot;[^&]*?&quot;)(?!\s*:)", r'<span class="cc-str">\1</span>', line
    )
    line = re.sub(r"\b(true|false|null)\b", r'<span class="cc-kw">\1</span>', line)
    line = _TINT_NUM.sub(r'<span class="cc-num">\1</span>', line)
    return line


def _code_card(label: str, lang: str, lines: list[str], tint) -> str:
    """Assemble a code card from raw (un-escaped) code lines + a tinter."""
    rendered = "\n".join(tint(html.escape(ln)) for ln in lines)
    return (
        f'<div class="codecard">'
        f'<div class="codecard-head"><span class="codecard-dot"></span>'
        f'<span class="codecard-name">{html.escape(label)}</span>'
        f'<span class="codecard-lang">{html.escape(lang)}</span></div>'
        f'<pre class="codecard-body"><code>{rendered}</code></pre></div>'
    )


# a tcolorbox is a "code listing" when its body is one <p> of <code>..</code>
# runs joined by <br>; capture the inner <p> so we can rebuild it as a card.
_LISTING_RE = re.compile(
    r'<div class="tcolorbox">\s*<p>((?:\s*<code>.*?</code>\s*(?:<br\s*/?>)?)+)\s*</p>\s*</div>',
    flags=re.S,
)
_CODE_LINE_RE = re.compile(r"<code>(.*?)</code>", flags=re.S)


def code_cards(body: str) -> str:
    """Turn mangled code into clean light code cards.

    (1) Reward-spec / JSON tcolorboxes that pandoc rendered as runs of inline
        <code>..</code> joined by <br> become real YAML/JSON code cards.
    (2) The two flattened algorithm environments become pseudocode cards.
    """

    def listing(m: re.Match) -> str:
        raw = [html.unescape(c) for c in _CODE_LINE_RE.findall(m.group(1))]
        joined = "\n".join(raw)
        if "phases:" in joined or ("name:" in joined and "reward:" in joined):
            return _code_card("reward_spec.yaml", "yaml", raw, _tint_yaml_line)
        if joined.lstrip().startswith("{") or '"instance_id"' in joined:
            return _code_card("proc_score.json", "json", raw, _tint_json_line)
        # other code listings (none expected) — still render as a plain card
        return _code_card("listing", "text", raw, lambda s: s)

    body = _LISTING_RE.sub(listing, body)

    # (2) replace the two algorithm blocks, matched by distinctive content.
    body = re.sub(
        r'<div class="algorithm">\s*<div class="algorithmic">\s*<p>.*?all atoms appearing.*?</p>\s*</div>\s*</div>',
        _algo_card("Algorithm: Byte-Pair Encoding (action vocabulary)", ALGO_BPE),
        body,
        count=1,
        flags=re.S,
    )
    body = re.sub(
        r'<div class="algorithm">\s*<div class="algorithmic">\s*<p>.*?minimum support.*?</p>\s*</div>\s*</div>',
        _algo_card("Algorithm: PrefixSpan", ALGO_PREFIXSPAN),
        body,
        count=1,
        flags=re.S,
    )
    return body


def main() -> None:
    body = run_pandoc()
    body = fix_figures(body)
    body, jsd_data = inject_interactive(body)
    body = code_cards(body)
    body = margin_citations(body)
    body = glossary(body)
    toc = build_toc(body)
    abstract = extract_abstract()

    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(TITLE)} · ProcGrep</title>
<style>{STYLE}</style>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
</head><body>
<div class="progress" id="progress"></div>
<div class="topbar"><div class="in"><span class="mark">procgrep</span>
<nav><a href="#showcase">demo</a><a href="#refs">references</a>
<a href="explorer/">explorer ↗</a><a href="https://github.com/hamidahoderinwale/procgrep">github ↗</a></nav></div></div>

<div class="hero">
  <h1>{html.escape(TITLE)}</h1>
  <div class="sub">{html.escape(SUBTITLE)}</div>
  <div class="byline"><span>{html.escape(AUTHOR)}</span><span>Taste Labs</span>
    <a href="https://github.com/hamidahoderinwale/procgrep">ProcGrep on GitHub ↗</a></div>
</div>

<div class="abstract"><div class="lab">Abstract</div><p>{abstract}</p></div>

{SHOWCASE}

{toc}

<article>
{body}
</article>

{jsd_data}
<script>{JSD_JS}</script>
<script>{FT_JS}</script>
<script>{PIPE_JS}</script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script>window.addEventListener('load',function(){{
  if(!window.katex)return;
  document.querySelectorAll('span.math').forEach(function(el){{
    try{{katex.render(el.textContent,el,{{displayMode:el.classList.contains('display'),throwOnError:false}});}}catch(e){{}}
  }});
}});</script>
<script>
(function(){{const p=document.getElementById('progress');
function upd(){{const h=document.documentElement,max=h.scrollHeight-h.clientHeight;
p.style.width=(max>0?(h.scrollTop/max*100):0)+'%';}}
addEventListener('scroll',upd,{{passive:true}});addEventListener('resize',upd);upd();}})();
</script>
</body></html>
"""
    OUT.write_text(page)
    print(f"wrote {OUT}  ({len(page) // 1024} KB, body {len(body) // 1024} KB)")


if __name__ == "__main__":
    main()
