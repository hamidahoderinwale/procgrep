"""Figure: run-length distribution by interaction surface.

Three small-multiple histograms of agent run length (actions per human prompt)
on a shared log2 x-axis: Claude Code and Cursor (interactive, local sessions)
versus SWE-bench (autonomous, from the spine store). The shared log axis is
deliberate -- run length spans single digits to hundreds, so a linear axis
would squash one side. Emits a self-contained HTML; render to PNG for the essay.

Run lengths come from local sources (your own Claude Code + Cursor sessions and
data/procgrep_spines.parquet), so regenerate on the author's machine. Only the
binned counts reach the figure -- no prompt text or code.
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st
from pathlib import Path

from procgrep.ingest.adapters.claude_code import build_panel_session
from procgrep.ingest.adapters.cursor_vscdb import build_panel_sessions

PAL = {
    "paper": "#F7F5F2",
    "ink": "#14110E",
    "copper": "#CB4D20",
    "blue": "#5692E5",
    "olive": "#585E53",
    "rule": "#d9d4cc",
}
EDGES = [2**i for i in range(11)]  # 1,2,4,...,1024


def _lines(path: str) -> list:
    out = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def claude_code_runs(path: str = "~/.claude/projects", recent: int = 30) -> list[int]:
    base = Path(path).expanduser()
    files = sorted(glob.glob(str(base / "*" / "*.jsonl"))) + sorted(
        glob.glob(str(base / "*.jsonl"))
    )
    return [
        len(t["seq"])
        for f in files[-recent:]
        for t in build_panel_session({"events": _lines(f)})["turns"]
    ]


def cursor_runs(db: str, limit: int = 60) -> list[int]:
    return [
        len(t["seq"])
        for s in build_panel_sessions(Path(db).expanduser(), limit=limit)
        for t in s["turns"]
    ]


def autonomous_runs(parquet: str = "data/procgrep_spines.parquet", sample: int = 3000) -> list[int]:
    import pandas as pd

    df = pd.read_parquet(parquet)
    col = df["spine"].sample(min(sample, len(df)), random_state=0)
    return [len(str(s).split()) for s in col]


def _bin_index(run: int) -> int:
    return min(max(0, int(math.log2(max(1, run)))), len(EDGES) - 2)


def _fractions(runs: list[int]) -> list[float]:
    counts = [0] * (len(EDGES) - 1)
    for r in runs:
        counts[_bin_index(r)] += 1
    n = len(runs) or 1
    return [c / n for c in counts]


def _panel(x0: float, title: str, runs: list[int], ymax: float, color: str) -> str:
    pw, ph, mt = 268.0, 168.0, 30.0
    fr = _fractions(runs)
    nb = len(fr)
    bw = pw / nb
    bars = ""
    for i, f in enumerate(fr):
        if not f:
            continue
        h = (f / ymax) * ph
        bars += (
            f'<rect x="{x0 + i * bw + 0.6:.1f}" y="{mt + ph - h:.1f}" '
            f'width="{bw - 1.2:.1f}" height="{h:.1f}" fill="{color}"/>'
        )
    xt = ""
    for v in (1, 8, 64, 512):
        i = _bin_index(v)
        x = x0 + (i + 0.5) * bw
        xt += (
            f'<text x="{x:.1f}" y="{mt + ph + 15:.0f}" text-anchor="middle" '
            f'font-size="10" fill="{PAL["olive"]}">{v}</text>'
        )
    base = (
        f'<line x1="{x0:.1f}" y1="{mt + ph:.1f}" x2="{x0 + pw:.1f}" y2="{mt + ph:.1f}" '
        f'stroke="{PAL["rule"]}" stroke-width="1"/>'
    )
    head = (
        f'<text x="{x0:.1f}" y="18" font-size="12" font-weight="600" fill="{PAL["ink"]}">{title}</text>'
        f'<text x="{x0:.1f}" y="{mt + ph + 33:.0f}" font-size="10" fill="{PAL["olive"]}">'
        f"median {int(st.median(runs))} · mean {st.mean(runs):.0f} · n {len(runs)}</text>"
    )
    return head + bars + base + xt


def build_html(runs: dict[str, list[int]]) -> str:
    ymax = max(max(_fractions(r)) for r in runs.values())
    sw, sh = 960, 250
    colors = [PAL["copper"], PAL["blue"], PAL["olive"]]
    panels = "".join(
        _panel(20 + i * 312, name, runs[name], ymax, colors[i]) for i, name in enumerate(runs)
    )
    # one shared y label on the left
    yl = (
        f'<text x="14" y="34" font-size="10" fill="{PAL["olive"]}">{ymax * 100:.0f}%</text>'
        f'<text x="14" y="198" font-size="10" fill="{PAL["olive"]}">0</text>'
    )
    svg = (
        f'<svg viewBox="0 0 {sw} {sh}" width="100%" role="img" '
        f'aria-label="Run length by interaction surface">{yl}{panels}'
        f'<text x="{sw // 2}" y="{sh - 4}" text-anchor="middle" font-size="11" '
        f'fill="{PAL["olive"]}">actions per human prompt, log scale</text></svg>'
    )
    return (
        f"<title>procgrep · run length by surface</title>"
        f"<style>body{{margin:0 auto;max-width:1000px;background:{PAL['paper']};color:{PAL['ink']};"
        f"font:13px/1.5 ui-monospace,Menlo,monospace;padding:30px 20px}}"
        f".cap{{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:{PAL['olive']};margin:0 0 10px}}"
        f".chart{{background:#fff;border-radius:8px;padding:14px 16px}}</style>"
        f'<p class="cap">run length by interaction surface</p>'
        f'<div class="chart">{svg}</div>'
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cursor", default="~/Library/Application Support/Cursor/User/globalStorage/state.vscdb"
    )
    parser.add_argument("--out", default="docs/figures/interaction_surface_runlength.html")
    args = parser.parse_args()
    runs = {
        "Claude Code": claude_code_runs(),
        "Cursor": cursor_runs(args.cursor),
        "SWE-bench autonomous": autonomous_runs(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(runs))
    print("wrote", out)
    for name, r in runs.items():
        print(f"  {name}: n={len(r)} median={int(st.median(r))} mean={st.mean(r):.1f}")


if __name__ == "__main__":
    main()
