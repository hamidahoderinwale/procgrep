"""Multi-agent fix-type comparison: table + Cleveland dot plot.

Reads patch_analysis_*.jsonl files and produces:
  1. Printed summary table (fix type breakdown + pass rates per agent)
  2. fig_patch_types.png, a Cleveland dot plot: pass rate by fix type and agent
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from scripts.theme import AGENT_COLORS, GRAY, NEAR_BLACK, register

register()

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = RES / "paper_figures"
OUT.mkdir(exist_ok=True)

AGENT_FILES = {
    "Claude-3 Opus": "patch_analysis_claude3opus.jsonl",
    "Claude-3.5 Sonnet": "patch_analysis_claude3.5sonnet.jsonl",
    "Claude-4 Sonnet": "patch_analysis_claude4sonnet.jsonl",
    "SWE-agent-LM-32B": "patch_analysis.jsonl",
}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()]


def fix_type(r: dict) -> str:
    if r.get("test_only"):
        return "test-only"
    if r.get("no_patch"):
        return "no patch"
    if r.get("n_test_files", 0) > 0 and r.get("n_source_files", 0) > 0:
        return "source + tests"
    if r.get("n_source_files", 0) > 0:
        return "source only"
    return "no patch"


def summarise(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r.get("resolved") is not None]
    if not labeled:
        return {}
    types = [fix_type(r) for r in labeled]
    resolved = [r["resolved"] for r in labeled]
    n = len(labeled)

    out = {"n_total": n, "n_pass": sum(resolved), "pass_rate": np.mean(resolved)}
    for ft in ["source only", "source + tests", "test-only", "no patch"]:
        idx = [i for i, t in enumerate(types) if t == ft]
        out[f"n_{ft}"] = len(idx)
        out[f"pct_{ft}"] = len(idx) / n
        if idx:
            out[f"pass_rate_{ft}"] = np.mean([resolved[i] for i in idx])
        else:
            out[f"pass_rate_{ft}"] = float("nan")
    return out


def print_table(stats: dict[str, dict]):
    FIX_TYPES = ["source only", "source + tests", "test-only"]
    print("=" * 100)
    print("FIX-TYPE BREAKDOWN BY AGENT")
    print("=" * 100)
    print(
        f"  {'Agent':22s}  {'n':>5s}  {'Pass%':>6s}  " + "  ".join(f"{ft:>16s}" for ft in FIX_TYPES)
    )
    print("  " + "-" * 95)
    for agent, s in stats.items():
        if not s:
            print(f"  {agent:22s}  (no labeled data)")
            continue
        row = f"  {agent:22s}  {s['n_total']:>5d}  {s['pass_rate']:>5.1%}  "
        for ft in FIX_TYPES:
            n_ft = s.get(f"n_{ft}", 0)
            pct_ft = s.get(f"pct_{ft}", 0)
            pr_ft = s.get(f"pass_rate_{ft}", float("nan"))
            if n_ft == 0:
                row += f"  {'—':>16s}"
            else:
                pr_str = f"{pr_ft:.0%}" if not np.isnan(pr_ft) else "—"
                row += f"  {n_ft:3d}({pct_ft:.0%}) {pr_str:>4s}"
        print(row)
    print()

    print("Pass rates by fix type:")
    print(f"  {'Fix type':20s}" + "".join(f"  {a[:18]:>18s}" for a in stats))
    print("  " + "-" * (20 + 20 * len(stats)))
    for ft in FIX_TYPES:
        row = f"  {ft:20s}"
        for s in stats.values():
            pr = s.get(f"pass_rate_{ft}", float("nan")) if s else float("nan")
            row += f"  {f'{pr:.1%}' if not np.isnan(pr) else '—':>18s}"
        print(row)
    print()


def dot_plot(stats: dict[str, dict]):
    FIX_TYPES = ["source only", "source + tests"]
    rows = []
    for agent, s in stats.items():
        if not s:
            continue
        for ft in FIX_TYPES:
            pr = s.get(f"pass_rate_{ft}", float("nan"))
            n = s.get(f"n_{ft}", 0)
            if np.isnan(pr) or n < 5:
                continue
            rows.append(
                {
                    "agent": agent,
                    "fix_type": ft,
                    "pass_rate": pr,
                    "n": n,
                }
            )
    df = pd.DataFrame(rows)

    agent_order = list(stats.keys())
    color_map = {a: AGENT_COLORS.get(a, GRAY) for a in agent_order}

    dots = (
        alt.Chart(df)
        .mark_point(filled=True, size=100)
        .encode(
            x=alt.X(
                "pass_rate:Q",
                title="Pass rate",
                scale=alt.Scale(domain=[0, 0.75]),
                axis=alt.Axis(format="%", grid=True, gridColor="#eeeeee"),
            ),
            y=alt.Y("agent:N", sort=agent_order, title=None),
            color=alt.Color(
                "agent:N",
                scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())),
                legend=None,
            ),
            tooltip=[
                "agent",
                "fix_type",
                alt.Tooltip("pass_rate:Q", format=".1%"),
                alt.Tooltip("n:Q", title="n trajectories"),
            ],
        )
    )

    # detail="agent" draws one connecting segment per agent
    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=1, color=GRAY)
        .encode(
            x="pass_rate:Q",
            y=alt.Y("agent:N", sort=agent_order),
            detail="agent:N",
        )
    )

    labels = (
        alt.Chart(df)
        .mark_text(dy=-10, fontSize=8, color=NEAR_BLACK)
        .encode(
            x="pass_rate:Q",
            y=alt.Y("agent:N", sort=agent_order),
            text=alt.Text("fix_type:N"),
        )
    )

    chart = (lines + dots + labels).properties(
        title="Pass rate by fix type and agent",
        width=380,
        height=220,
    )

    path = OUT / "fig_patch_types.png"
    chart.save(str(path), scale_factor=2)
    print(f"Saved: {path.name}")
    return path


if __name__ == "__main__":
    stats = {}
    for agent, fname in AGENT_FILES.items():
        p = RES / fname
        if not p.exists():
            print(f"  MISSING: {fname}")
            stats[agent] = {}
            continue
        rows = load(p)
        stats[agent] = summarise(rows)
        print(f"  Loaded {len(rows)} rows for {agent}")

    print_table(stats)
    dot_plot(stats)
