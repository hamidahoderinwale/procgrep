"""Generate all paper figures for procgrep.

Run from /Users/hamidaho/learning-from-dev/procgrep-audits/
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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
OUT.mkdir(parents=True, exist_ok=True)

## helpers

SWEBENCH_AGENTS = [  # SWE-agent format (same scaffold = clean comparison)
    "Claude-3 Opus",
    "Claude-3.5 Sonnet",
    "Claude-4 Sonnet",
    "SWE-agent-LM-32B",
    "GPT-4",
    "GPT-4o",
]
ALL_AGENTS = SWEBENCH_AGENTS + ["Claude-4.5 Sonnet", "GPT-5"]

FINGERPRINT_FILES = {
    # Use n500 files where available; fall back to smaller pulls for newer models
    "Claude-3 Opus": "fingerprints_claude3opus_n500.jsonl",
    "Claude-3.5 Sonnet": "fingerprints_claude3.5sonnet_n500.jsonl",
    "Claude-4 Sonnet": "fingerprints_claude4sonnet_n500.jsonl",
    "Claude-4.5 Sonnet": "fingerprints_claude4.5sonnet.jsonl",
    "SWE-agent-LM-32B": "fingerprints_child_n500.jsonl",
    "GPT-4": "fingerprints_gpt4_n500.jsonl",
    "GPT-4o": "fingerprints_gpt4o_n500.jsonl",
    "GPT-5": "fingerprints_gpt5.jsonl",
}

RICH_FILES = {
    "Claude-3 Opus": "rich_features_20240402_sweagent_claude3opus.jsonl",
    "Claude-3.5 Sonnet": "rich_features_20240620_sweagent_claude3_5son.jsonl",
    "Claude-4 Sonnet": "rich_features_20250522_sweagent_claude_4_son.jsonl",
    "SWE-agent-LM-32B": "rich_features_20250511_sweagent_lm_32b.jsonl",
    "GPT-4": "rich_features_20240402_sweagent_gpt4.jsonl",
    "GPT-4o": "rich_features_20240728_sweagent_gpt4o.jsonl",
}

FAMILY = {
    "Claude-3 Opus": "Anthropic",
    "Claude-3.5 Sonnet": "Anthropic",
    "Claude-4 Sonnet": "Anthropic",
    "Claude-4.5 Sonnet": "Anthropic",
    "SWE-agent-LM-32B": "SFT-distilled",
    "GPT-4": "OpenAI",
    "GPT-4o": "OpenAI",
    "GPT-5": "OpenAI",
}

ERA = {
    "Claude-3 Opus": "2024 Q1",
    "Claude-3.5 Sonnet": "2024 Q2",
    "GPT-4": "2024 Q1",
    "GPT-4o": "2024 Q3",
    "Claude-4 Sonnet": "2025 Q2",
    "SWE-agent-LM-32B": "2025 Q2",
    "Claude-4.5 Sonnet": "2025 Q4",
    "GPT-5": "2025 Q3",
}

SCAFFOLD = {
    "Claude-3 Opus": "SWE-agent",
    "Claude-3.5 Sonnet": "SWE-agent",
    "Claude-4 Sonnet": "SWE-agent",
    "SWE-agent-LM-32B": "SWE-agent",
    "GPT-4": "SWE-agent",
    "GPT-4o": "SWE-agent",
    "Claude-4.5 Sonnet": "OpenHands",
    "GPT-5": "OpenHands",
}

W, H = 560, 380  # base figure dims


def load_fps(agents=None, layer="canonical", cap=9999):
    agents = agents or SWEBENCH_AGENTS
    rows = []
    for name in agents:
        fp = RES / FINGERPRINT_FILES.get(name, "")
        if not fp.exists():
            continue
        for i, line in enumerate(fp.read_text().splitlines()):
            if i >= cap:
                break
            d = json.loads(line)
            atoms = d.get(f"atoms_{layer}", [])
            if atoms:
                rows.append(
                    {
                        "agent": name,
                        "instance_id": d["instance_id"],
                        "atoms": atoms,
                        "resolved": d.get("resolved"),
                        "n_steps": d.get("n_steps", 0),
                    }
                )
    return rows


def atom_freqs(rows, layer="canonical"):
    """Return DataFrame with agent, atom, pct (action-only, no think)."""
    data = []
    for r in rows:
        atoms = [a for a in r["atoms"] if a != "think"]
        total = max(1, len(atoms))
        cnt = Counter(atoms)
        for atom, c in cnt.items():
            data.append({"agent": r["agent"], "atom": atom, "count": c, "total": total})
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df = df.groupby(["agent", "atom"]).agg({"count": "sum", "total": "sum"}).reset_index()
    df["pct"] = 100 * df["count"] / df["total"]
    return df


def save(chart, name):
    path = OUT / f"{name}.png"
    chart.save(str(path), scale_factor=2)
    print(f"  saved {name}.png")
    return path


## Figure 1: JSD Heatmaps (canonical + native side-by-side)


def fig_jsd_heatmaps():
    # Use n=500 data if available, fall back to n=50 summary
    canon_path = RES / "jsd_canonical_n500.json"
    native_path = RES / "jsd_native_n500.json"
    if canon_path.exists() and native_path.exists():
        canon_data = json.loads(canon_path.read_text())
        native_data = json.loads(native_path.read_text())
        data_src = {"jsd_canonical": canon_data, "jsd_native": native_data}
    else:
        data_src = json.loads((RES / "multi_agent_analysis/summary.json").read_text())

    def make_hm(data, title, domain_max=0.65):
        recs = data["records"]
        df = pd.DataFrame(recs)
        order = [
            "Claude-3 Opus",
            "GPT-4",
            "GPT-4o",
            "Claude-3.5 Sonnet",
            "Claude-4 Sonnet",
            "SWE-agent-LM-32B",
        ]
        df["row"] = pd.Categorical(df["row"], categories=order, ordered=True)
        df["col"] = pd.Categorical(df["col"], categories=order, ordered=True)

        base = alt.Chart(df).encode(
            x=alt.X("col:O", sort=order, title=None, axis=alt.Axis(labelAngle=-35, labelLimit=120)),
            y=alt.Y("row:O", sort=order, title=None),
        )
        hm = base.mark_rect().encode(
            color=alt.Color(
                "jsd:Q",
                scale=alt.Scale(scheme="reds", domain=[0, domain_max]),
                legend=alt.Legend(title="JSD", gradientLength=80),
            )
        )
        txt = base.mark_text(fontSize=9).encode(
            text=alt.Text("jsd:Q", format=".2f"),
            color=alt.condition(
                alt.datum.jsd > domain_max * 0.5, alt.value("white"), alt.value(NEAR_BLACK)
            ),
        )
        return (hm + txt).properties(title=title, width=240, height=240)

    n_canon = canon_data.get("n_per_agent", "?") if canon_path.exists() else "50"
    chart = alt.hconcat(
        make_hm(data_src["jsd_canonical"], f"Canonical JSD (n≥{n_canon}/agent)", 0.60),
        make_hm(data_src["jsd_native"], f"Native JSD (n≥{n_canon}/agent)", 0.95),
        spacing=20,
    ).properties(title="Pairwise divergence in agent action sequences")

    return save(chart, "fig1_jsd_heatmaps")


## Figure 2: Canonical atom distributions


def fig_canonical_atoms():
    rows = load_fps(ALL_AGENTS, "canonical", cap=50)
    df = atom_freqs(rows, "canonical")
    if df.empty:
        print("  skipping fig2 — no data")
        return

    ATOM_ORDER = [
        "search_repo",
        "read_file",
        "edit",
        "run_test",
        "create_file",
        "delete_file",
        "submit",
        "other",
    ]
    ATOM_LABELS = {
        "search_repo": "search",
        "read_file": "read",
        "edit": "edit",
        "run_test": "test",
        "create_file": "create",
        "delete_file": "delete",
        "submit": "submit",
        "other": "other",
    }
    df = df[df["atom"].isin(ATOM_ORDER)].copy()
    df["atom_label"] = df["atom"].map(ATOM_LABELS)
    df["family"] = df["agent"].map(FAMILY)
    df["color"] = df["agent"].map(AGENT_COLORS)

    agent_order = ALL_AGENTS
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "pct:Q",
                title="% of actions",
            ),
            y=alt.Y("agent:N", sort=agent_order, title=None),
            color=alt.Color(
                "atom_label:N",
                scale=alt.Scale(
                    domain=[
                        "search",
                        "read",
                        "edit",
                        "test",
                        "create",
                        "delete",
                        "submit",
                        "other",
                    ],
                    range=[
                        "#5B9BD5",
                        "#2E75B6",
                        "#203864",
                        "#70AD47",
                        "#A9D18E",
                        "#FFC000",
                        "#FF0000",
                        "#C9C9C9",
                    ],
                ),
                legend=alt.Legend(title="action type", orient="bottom", columns=4),
            ),
            order=alt.Order("atom_label:N"),
            tooltip=["agent", "atom_label", alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(title="Action distribution per agent", width=W - 60, height=300)
    )
    return save(chart, "fig2_canonical_atoms")


## Figure 3: Native top-12 vocabulary heat-strip


def fig_native_heatstrip():
    rows = load_fps(SWEBENCH_AGENTS, "native", cap=50)
    data = []
    for r in rows:
        atoms = [a for a in r["atoms"] if a != "think"]
        total = max(1, len(atoms))
        for a, c in Counter(atoms).items():
            data.append({"agent": r["agent"], "tag": a, "count": c, "total": total})
    df = (
        pd.DataFrame(data)
        .groupby(["agent", "tag"])
        .agg({"count": "sum", "total": "sum"})
        .reset_index()
    )
    df["pct"] = 100 * df["count"] / df["total"]

    # Top 14 tags globally by total mass (exclude think)
    top_tags = df.groupby("tag")["count"].sum().sort_values(ascending=False).head(14).index.tolist()
    df = df[df["tag"].isin(top_tags)]
    df["tag_short"] = df["tag"].str.replace("str_replace_editor:", "sre:", regex=False)
    df["tag_short"] = df["tag_short"].str.replace("python:", "", regex=False)
    df["tag_short"] = df["tag_short"].str.replace("other:", "", regex=False)

    agent_order = SWEBENCH_AGENTS
    tag_order = df.groupby("tag_short")["pct"].mean().sort_values(ascending=False).index.tolist()

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(
                "tag_short:N",
                sort=tag_order,
                title=None,
                axis=alt.Axis(labelAngle=-40, labelLimit=100),
            ),
            y=alt.Y("agent:N", sort=agent_order, title=None),
            color=alt.Color(
                "pct:Q",
                scale=alt.Scale(scheme="blues", domain=[0, 30]),
                legend=alt.Legend(title="% actions"),
            ),
            tooltip=["agent", "tag_short", alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(title="Native vocabulary per agent, top 14 tags", width=W, height=260)
    )

    return save(chart, "fig3_native_heatstrip")


## Figure 4: Procgrep metrics (effective vocab + entropy + seq length)


def fig_procgrep_metrics():
    import sys as _sys

    _sys.path.insert(0, "/Users/hamidaho/learning-from-dev/procgrep/src")
    from procgrep.bpe import fit_bpe
    from procgrep.encode import encode
    from procgrep.stats import effective_vocab_size_per_group, entropies_per_group
    from procgrep.types import Trace

    rows = load_fps(SWEBENCH_AGENTS, "canonical", cap=50)
    traces = [Trace(trace_id=r["instance_id"], agent=r["agent"], atoms=r["atoms"]) for r in rows]
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=64, seed=0)
    fps = encode(traces, vocab=vocab)
    evs = effective_vocab_size_per_group(fps, group_by="agent")
    ents = entropies_per_group(fps, group_by="agent")
    orig = defaultdict(list)
    for r in rows:
        orig[r["agent"]].append(len([a for a in r["atoms"] if a != "think"]))

    data = []
    for name in SWEBENCH_AGENTS:
        ev = evs.get(name, 0)
        en = getattr(ents.get(name), "median", 0)
        seqlen = sum(orig[name]) / max(1, len(orig[name]))
        data.append(
            {
                "agent": name,
                "eff_vocab": ev,
                "entropy": en,
                "seq_len": seqlen,
                "family": FAMILY[name],
                "color": AGENT_COLORS[name],
            }
        )
    df = pd.DataFrame(data)
    agent_order = SWEBENCH_AGENTS

    def metric_bar(metric, title, fmt=".1f"):
        return (
            alt.Chart(df)
            .mark_bar(size=18)
            .encode(
                x=alt.X(f"{metric}:Q", title=title),
                y=alt.Y(
                    "agent:N",
                    sort=agent_order,
                    title=None,
                    axis=alt.Axis(labels=(metric == "eff_vocab")),
                ),
                color=alt.Color(
                    "agent:N",
                    scale=alt.Scale(
                        domain=list(AGENT_COLORS.keys()), range=list(AGENT_COLORS.values())
                    ),
                    legend=None,
                ),
                tooltip=["agent", alt.Tooltip(f"{metric}:Q", format=fmt)],
            )
            .properties(width=160, height=230, title=title)
        )

    chart = alt.hconcat(
        metric_bar("eff_vocab", "Effective procedures"),
        metric_bar("entropy", "Entropy (bits)", ".2f"),
        metric_bar("seq_len", "Avg actions / trace", ".0f"),
        spacing=12,
    ).properties(title="Procedural fingerprint metrics per agent")

    return save(chart, "fig4_procgrep_metrics")


## Figure 5: Identification probe F1


def fig_probe_f1():
    # Use the stratified k-fold probe file, NOT the multi_agent summary (which has LOGO zeros)
    probe = json.loads((RES / "identification_probe_v1/summary.json").read_text())
    rows = []
    for layer, section in [("Canonical", probe["canonical"]), ("Native", probe["native"])]:
        for agent, f1 in section["per_agent_f1"].items():
            rows.append({"agent": agent, "layer": layer, "f1": float(f1)})
    df = pd.DataFrame(rows)
    agent_order = sorted(df["agent"].unique(), key=lambda a: -df[df["agent"] == a]["f1"].max())

    baseline = 1 / len(df["agent"].unique())

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("f1:Q", title="F1 score", scale=alt.Scale(domain=[0, 1.0])),
            y=alt.Y("agent:N", sort=agent_order, title=None),
            color=alt.Color(
                "layer:N",
                scale=alt.Scale(domain=["Canonical", "Native"], range=["#5B9BD5", "#203864"]),
                legend=alt.Legend(title="Alphabet", orient="bottom", direction="horizontal"),
            ),
            xOffset="layer:N",
            tooltip=["agent", "layer", alt.Tooltip("f1:Q", format=".2f")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [baseline]}))
        .mark_rule(color=GRAY, strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    return save(
        (chart + rule).properties(title="Agent identification F1", width=W - 60, height=300),
        "fig5_probe_f1",
    )


## Figure 6: Token verbosity + cost


def fig_tokens_cost():
    data = []
    for name in SWEBENCH_AGENTS:
        fp = RES / RICH_FILES.get(name, "")
        if not fp.exists():
            continue
        rows = [json.loads(l) for l in fp.read_text().splitlines()]
        tpa = sum(r.get("tokens_per_action", 0) for r in rows) / max(1, len(rows))
        cost = sum(r.get("instance_cost", 0) for r in rows) / max(1, len(rows))
        data.append(
            {
                "agent": name,
                "tokens_per_action": tpa,
                "cost_per_trace": cost,
                "family": FAMILY[name],
            }
        )
    df = pd.DataFrame(data)
    agent_order = SWEBENCH_AGENTS

    def bar_with_labels(metric, title, fmt=".0f", prefix=""):
        b = (
            alt.Chart(df)
            .mark_bar(size=18)
            .encode(
                x=alt.X(f"{metric}:Q", title=title),
                y=alt.Y(
                    "agent:N",
                    sort=agent_order,
                    title=None,
                    axis=alt.Axis(labels=(metric == "tokens_per_action")),
                ),
                color=alt.Color(
                    "agent:N",
                    scale=alt.Scale(
                        domain=list(AGENT_COLORS.keys()), range=list(AGENT_COLORS.values())
                    ),
                    legend=None,
                ),
                tooltip=["agent", alt.Tooltip(f"{metric}:Q", format=fmt)],
            )
        )
        lbl = (
            alt.Chart(df)
            .mark_text(align="left", dx=4, fontSize=9, color=NEAR_BLACK)
            .encode(
                x=alt.X(f"{metric}:Q"),
                y=alt.Y("agent:N", sort=agent_order),
                text=alt.Text(f"{metric}:Q", format=fmt),
            )
        )
        return (b + lbl).properties(width=220, height=230, title=title)

    chart = alt.hconcat(
        bar_with_labels("tokens_per_action", "Tokens per action"),
        bar_with_labels("cost_per_trace", "Cost per trace (USD)", ".2f"),
        spacing=16,
    ).properties(title="Verbosity and cost per agent")

    return save(chart, "fig6_tokens_cost")


## Figure 7: File consumption


def fig_files():
    data = []
    for name in SWEBENCH_AGENTS:
        fp = RES / RICH_FILES.get(name, "")
        if not fp.exists():
            continue
        rows = [json.loads(l) for l in fp.read_text().splitlines()]
        nr = sum(r.get("n_files_read", 0) for r in rows) / max(1, len(rows))
        ne = sum(r.get("n_files_edited", 0) for r in rows) / max(1, len(rows))
        data.append({"agent": name, "type": "read", "files": nr})
        data.append({"agent": name, "type": "edited", "files": ne})
    df = pd.DataFrame(data)
    agent_order = SWEBENCH_AGENTS

    chart = (
        alt.Chart(df)
        .mark_bar(size=22)
        .encode(
            x=alt.X("files:Q", title="Files per trace"),
            y=alt.Y("agent:N", sort=agent_order, title=None),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=["read", "edited"], range=["#AED6F1", "#1A5276"]),
                legend=alt.Legend(title=None),
            ),
            xOffset="type:N",
            tooltip=["agent", "type", alt.Tooltip("files:Q", format=".1f")],
        )
        .properties(title="Files explored per trace", width=W - 100, height=260)
    )

    return save(chart, "fig7_files")


## Figure 8: Shell command taxonomy


def fig_shell_taxonomy():
    def categorise(tag: str) -> str:
        if tag == "think":
            return None
        if tag in ("str_replace_editor:view", "ls", "cat"):
            return "read (editor)"
        if tag in (
            "str_replace_editor:str_replace",
            "str_replace_editor:insert",
            "str_replace_editor:undo",
            "edit",
        ):
            return "edit (editor)"
        if tag in ("str_replace_editor:create", "mkdir_touch", "echo_redirect"):
            return "create (editor)"
        if tag in ("pipe:find", "find", "pipe:grep", "grep", "search_dir", "search_file"):
            return "search/find"
        if tag in ("python:script", "python:pytest", "python:pytest"):
            return "run (python)"
        if tag == "submit":
            return "submit"
        if tag in ("rm", "delete_file"):
            return "delete"
        if tag.startswith("other:git"):
            return "git"
        if tag.startswith("other:pip"):
            return "pip"
        if tag.startswith("other:sed"):
            return "sed"
        if tag in ("other:exit_cost", "other:exit_format"):
            return "artifact"
        if tag.startswith("other:"):
            return "other cmd"
        return "other cmd"

    data = []
    for name in ALL_AGENTS:
        fp = RES / FINGERPRINT_FILES.get(name, "")
        if not fp.exists():
            continue
        rows = [json.loads(l) for l in fp.read_text().splitlines()[:50]]
        cnt = Counter()
        for r in rows:
            for a in r.get("atoms_native", []):
                cat = categorise(a)
                if cat:
                    cnt[cat] += 1
        total = max(1, sum(cnt.values()))
        for cat, c in cnt.items():
            data.append({"agent": name, "category": cat, "pct": 100 * c / total})
    df = pd.DataFrame(data)

    CAT_ORDER = [
        "search/find",
        "read (editor)",
        "edit (editor)",
        "run (python)",
        "create (editor)",
        "delete",
        "submit",
        "git",
        "pip",
        "sed",
        "artifact",
        "other cmd",
    ]
    CAT_COLORS = {
        "search/find": "#5B9BD5",
        "read (editor)": "#2E75B6",
        "edit (editor)": "#203864",
        "run (python)": "#70AD47",
        "create (editor)": "#A9D18E",
        "delete": "#FF0000",
        "submit": "#7B7B7B",
        "git": "#FF8C00",
        "pip": "#C55A11",
        "sed": "#9E480E",
        "artifact": "#C9C9C9",
        "other cmd": "#E2E2E2",
    }

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                "pct:Q",
                stack="normalize",
                title="% of actions (normalized)",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("agent:N", sort=ALL_AGENTS, title=None),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=CAT_ORDER, range=[CAT_COLORS[c] for c in CAT_ORDER]),
                legend=alt.Legend(title="action category", columns=2, orient="bottom"),
            ),
            order=alt.Order("category:N"),
            tooltip=["agent", "category", alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(title="Action taxonomy per agent", width=W, height=310)
    )

    return save(chart, "fig8_shell_taxonomy")


## Figure 9: OOD score by pass / fail


def fig_ood():
    ood_path = RES / "ood_analysis_v1/ood_scores_canonical.jsonl"
    if not ood_path.exists():
        print("  skipping fig9 — no OOD data")
        return
    rows = [json.loads(l) for l in ood_path.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df = df[df["agent"] == "SWE-agent-LM-32B"].copy()
    df = df[df["resolved"].notna()].copy()
    df["outcome"] = df["resolved"].apply(lambda r: "pass" if r else "fail")

    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.65, binSpacing=1)
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=20), title="Intra-agent OOD score"),
            y=alt.Y("count():Q", title="Trajectories"),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=["pass", "fail"], range=["#1A7A4A", "#B03A2E"]),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(title="OOD score distribution by outcome", width=W - 80, height=270)
    )
    return save(chart, "fig9_ood_scores")


## Figure 10: Within-trajectory drift


def fig_drift():
    rows = load_fps(SWEBENCH_AGENTS, "canonical", cap=50)
    data = []
    for r in rows:
        atoms = [a for a in r["atoms"] if a != "think"]
        n = len(atoms)
        if n < 6:
            continue
        half = n // 2
        early = Counter(atoms[:half])
        late = Counter(atoms[half:])
        for atom in set(list(early) + list(late)):
            ep = 100 * early.get(atom, 0) / max(1, half)
            lp = 100 * late.get(atom, 0) / max(1, n - half)
            data.append({"agent": r["agent"], "atom": atom, "window": "early", "pct": ep})
            data.append({"agent": r["agent"], "atom": atom, "window": "late", "pct": lp})
    df = pd.DataFrame(data).groupby(["agent", "atom", "window"])["pct"].mean().reset_index()
    df = df[df["atom"].isin(["search_repo", "read_file", "edit", "run_test", "create_file"])]

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("window:N", sort=["early", "late"], title=None),
            y=alt.Y("pct:Q", title="% of actions"),
            color=alt.Color(
                "atom:N",
                scale=alt.Scale(
                    domain=["search_repo", "read_file", "edit", "run_test", "create_file"],
                    range=["#5B9BD5", "#2E75B6", "#203864", "#70AD47", "#A9D18E"],
                ),
                legend=alt.Legend(title="action type"),
            ),
            facet=alt.Facet("agent:N", sort=SWEBENCH_AGENTS, columns=3),
        )
        .properties(width=130, height=110)
        .properties(title="Within-trajectory drift, early vs late")
    )

    return save(chart, "fig10_drift")


## Figure 11a: Agent identity step patterns


def _is_repetition_of(longer: list[str], shorter: list[str]) -> bool:
    """True if `longer` is `shorter` repeated (possibly with a partial tail).

    Catches both exact cyclic repetition AND partial extensions like A + suffix(A),
    by checking whether `longer` is a prefix of `shorter` repeated indefinitely.
    """
    n = len(shorter)
    if n == 0 or len(longer) < n:
        return False
    # longer must match shorter[i % n] at every position
    return all(longer[i] == shorter[i % n] for i in range(len(longer)))


def _deduplicate_primitives(procs: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Keep only non-repetitive primitive procedures.

    Returns (deduplicated list, dict mapping kept procedure → number of
    redundant repetition-variants collapsed into it).
    """
    sorted_procs = sorted(procs, key=lambda p: len(p["procedure"].split("→")))
    kept: list[dict] = []
    variant_counts: dict[str, int] = {}

    for p in sorted_procs:
        steps = [s.strip() for s in p["procedure"].split("→")]
        # Check if this is a pure repetition of any already-kept primitive
        primitive_key = next(
            (
                k["procedure"]
                for k in kept
                if _is_repetition_of(steps, [s.strip() for s in k["procedure"].split("→")])
            ),
            None,
        )
        if primitive_key is not None:
            variant_counts[primitive_key] = variant_counts.get(primitive_key, 1) + 1
        else:
            kept.append(p)
            variant_counts[p["procedure"]] = 1

    return kept, variant_counts


def _shorten_proc(proc: str, max_steps: int = 8) -> str:
    """Abbreviate long procedure strings to at most max_steps atoms."""
    steps = [s.strip() for s in proc.split("→")]
    if len(steps) <= max_steps:
        return proc
    unique = list(dict.fromkeys(steps))
    if len(unique) == 1:
        return f"{unique[0]} ×{len(steps)}"
    if len(unique) <= 2:
        counts = Counter(steps)
        parts = [f"{s}×{counts[s]}" if counts[s] > 1 else s for s in unique]
        return " → ".join(parts)
    return " → ".join(steps[:max_steps]) + f" … ({len(steps)} steps)"


def fig_discriminative_agents():
    """Procedures that identify each agent group, computed from BPE on n500 data."""
    src = RES / "discriminative_procedures_n500.json"
    if not src.exists():
        print("    discriminative_procedures_n500.json missing — skipping fig11a")
        return None

    raw = json.loads(src.read_text())["canonical"]

    DISPLAY = {
        "Claude-4 Sonnet": ("Claude-4 Sonnet", "#5DADE2", "claude4_vs_old"),
        "old-cluster": ("2024-era models", "#7B241C", "claude4_vs_old"),
        "SWE-agent-LM-32B": ("SWE-agent-LM-32B", "#1A7A4A", "swe_lm_vs_claude4"),
    }

    rows = []
    for agent_key, (label, color, comp_key) in DISPLAY.items():
        procs = raw.get(comp_key, {}).get(agent_key, [])
        # totally exclusive, 2 to 8 steps: single atoms are BPE artifacts;
        # longer entries are partial extensions of shorter primitives
        excl = sorted(
            [
                p
                for p in procs
                if p["p_b"] == 0.0
                and 2 <= len(p["procedure"].split("→")) <= 8
                and p["p_a"] >= 0.002
            ],
            key=lambda p: -p["p_a"],
        )
        primitives, vcounts = _deduplicate_primitives(excl)
        # Count how many longer variants (all lengths) exist for each primitive
        all_excl = [p for p in procs if p["p_b"] == 0.0]
        for p in primitives[:3]:
            prim_steps = [s.strip() for s in p["procedure"].split("→")]
            n_longer = sum(
                1
                for q in all_excl
                if len(q["procedure"].split("→")) > len(prim_steps)
                and _is_repetition_of([s.strip() for s in q["procedure"].split("→")], prim_steps)
            )
            label_str = _shorten_proc(p["procedure"])
            if n_longer > 0:
                label_str += f"  (+{n_longer} longer)"
            rows.append(
                {
                    "group": label,
                    "procedure": label_str,
                    "freq": p["p_a"],
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("freq", ascending=False)
    COLORS = {
        "Claude-4 Sonnet": "#5DADE2",
        "2024-era models": "#7B241C",
        "SWE-agent-LM-32B": "#1A7A4A",
    }
    proc_order = list(df["procedure"])

    dots = (
        alt.Chart(df)
        .mark_circle(size=120)
        .encode(
            x=alt.X(
                "freq:Q",
                title="Frequency in agent (fraction of trajectories)",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 0.1]),
            ),
            y=alt.Y(
                "procedure:N",
                sort=proc_order,
                title=None,
                axis=alt.Axis(labelLimit=280, labelFontSize=9),
            ),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=list(COLORS.keys()), range=list(COLORS.values())),
                legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
            ),
            tooltip=[
                "group",
                "procedure",
                alt.Tooltip("freq:Q", format=".1%", title="freq in agent"),
            ],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=8, fontSize=8, fontStyle="italic")
        .encode(
            x="freq:Q",
            y=alt.Y("procedure:N", sort=proc_order),
            text="group:N",
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=list(COLORS.keys()), range=list(COLORS.values())),
                legend=None,
            ),
        )
    )
    return save(
        (dots + labels).properties(
            title="Habits exclusive to each agent group (never appear in others)",
            width=420,
            height=240,
        ),
        "fig11a_discriminative_agents",
    )


## Figure 11b: Pass / fail step patterns


def fig_discriminative_outcomes():
    """Procedures that predict pass vs fail, computed from BPE on n500 data."""
    src = RES / "discriminative_procedures_n500.json"
    if not src.exists():
        print("    discriminative_procedures_n500.json missing — skipping fig11b")
        return None

    raw = json.loads(src.read_text())["canonical"]["pass_vs_fail"]

    rows = []
    for outcome, label in [("fail", "Fail"), ("pass", "Pass")]:
        all_excl = [p for p in raw.get(outcome, []) if p["p_b"] == 0.0]
        excl = sorted(
            [p for p in all_excl if 2 <= len(p["procedure"].split("→")) <= 8 and p["p_a"] >= 0.002],
            key=lambda p: -p["p_a"],
        )
        primitives, _ = _deduplicate_primitives(excl)
        if not primitives:
            # Explicitly show the absence: pass has no exclusive stereotype
            rows.append(
                {
                    "outcome": label,
                    "procedure": "(no exclusive habit — varied approaches succeed)",
                    "freq": 0.0,
                }
            )
            continue
        for p in primitives[:3]:
            prim_steps = [s.strip() for s in p["procedure"].split("→")]
            n_longer = sum(
                1
                for q in all_excl
                if len(q["procedure"].split("→")) > len(prim_steps)
                and _is_repetition_of([s.strip() for s in q["procedure"].split("→")], prim_steps)
            )
            label_str = _shorten_proc(p["procedure"])
            if n_longer > 0:
                label_str += f"  (+{n_longer} longer)"
            rows.append(
                {
                    "outcome": label,
                    "procedure": label_str,
                    "freq": p["p_a"],
                }
            )

    df = pd.DataFrame(rows).sort_values("freq", ascending=False)
    proc_order = list(df["procedure"])

    dots = (
        alt.Chart(df)
        .mark_circle(size=130)
        .encode(
            x=alt.X(
                "freq:Q",
                title="Frequency in outcome group",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 0.007]),
            ),
            y=alt.Y(
                "procedure:N",
                sort=proc_order,
                title=None,
                axis=alt.Axis(labelLimit=340, labelFontSize=9),
            ),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=["Pass", "Fail"], range=["#1A7A4A", "#C0392B"]),
                legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
            ),
            tooltip=["outcome", "procedure", alt.Tooltip("freq:Q", format=".2%", title="freq")],
        )
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=8, fontSize=9, fontStyle="italic")
        .encode(
            x="freq:Q",
            y=alt.Y("procedure:N", sort=proc_order),
            text="outcome:N",
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(domain=["Pass", "Fail"], range=["#1A7A4A", "#C0392B"]),
                legend=None,
            ),
        )
    )
    return save(
        (dots + labels).properties(
            title="Habits exclusive to failing runs vs pass-group diversity", width=440, height=160
        ),
        "fig11b_discriminative_outcomes",
    )


## Figure 12: Tier 1B SODP scatter


def fig_tier1b():
    """JSD distribution by outcome-match for matched agent pairs.

    For each pair, shows the spread of per-instance procedural distances,
    split by whether both agents agreed on the outcome (same pass/fail)
    or disagreed. SODP instances (same outcome, high JSD) are the paper's
    core claim: procedure adds information that outcome labels discard.
    """
    src = RES / "tier1b_matched_pairs_v1.json"
    if not src.exists():
        print("    tier1b_matched_pairs_v1.json missing — skipping fig12")
        return None

    data = json.loads(src.read_text())

    # the JSON stores only aggregates; re-derive per-instance JSD from fingerprint files
    fp_files = {
        "Claude-3.5 Sonnet": RES / "fingerprints_claude3.5sonnet_n500.jsonl",
        "Claude-4 Sonnet": RES / "fingerprints_claude4sonnet_n500.jsonl",
        "GPT-4o": RES / "fingerprints_gpt4o_n500.jsonl",
        "SWE-agent-LM-32B": RES / "fingerprints_child_n500.jsonl",
    }
    EPS = 1e-9
    CANON = [
        "edit",
        "read_file",
        "run_test",
        "search_repo",
        "create_file",
        "delete_file",
        "think",
        "error",
        "other",
    ]

    def _dist(atoms):
        cnt = Counter(atoms)
        v = np.array([cnt.get(a, 0) + EPS for a in CANON], dtype=float)
        return v / v.sum()

    def _jsd(p, q):
        m = (p + q) / 2
        return float(np.sum(p * np.log(p / m) + q * np.log(q / m)) / 2)

    corpora = {}
    for name, path in fp_files.items():
        if path.exists():
            corpora[name] = {
                json.loads(l)["instance_id"]: json.loads(l) for l in path.read_text().splitlines()
            }

    FOCUS = [
        ("Claude-3.5 Sonnet", "SWE-agent-LM-32B"),
        ("Claude-4 Sonnet", "SWE-agent-LM-32B"),
        ("Claude-4 Sonnet", "Claude-3.5 Sonnet"),
    ]

    rows = []
    for na, nb in FOCUS:
        if na not in corpora or nb not in corpora:
            continue
        common = set(corpora[na]) & set(corpora[nb])
        for iid in common:
            ra, rb = corpora[na][iid], corpora[nb][iid]
            if ra.get("resolved") is None or rb.get("resolved") is None:
                continue
            pa = _dist(ra.get("atoms_canonical", []))
            pb = _dist(rb.get("atoms_canonical", []))
            j = _jsd(pa, pb)
            sa, sb = bool(ra["resolved"]), bool(rb["resolved"])
            if sa and sb:
                match = "both pass"
            elif not sa and not sb:
                match = "both fail"
            else:
                match = "disagree"
            rows.append({"pair": f"{na}\nvs {nb}", "jsd": j, "match": match})

    if not rows:
        return None

    df = pd.DataFrame(rows)
    PAIR_ORDER = [f"{a}\nvs {b}" for a, b in FOCUS if a in corpora and b in corpora]
    MATCH_COLORS = {
        "both pass": "#1A7A4A",
        "both fail": "#C0392B",
        "disagree": "#9E9E9E",
    }

    strip = (
        alt.Chart(df)
        .mark_circle(size=18, opacity=0.45)
        .encode(
            x=alt.X("jsd:Q", title="Procedural JSD", scale=alt.Scale(domain=[0, 0.5])),
            y=alt.Y("pair:N", sort=PAIR_ORDER, title=None, axis=alt.Axis(labelLimit=200)),
            color=alt.Color(
                "match:N",
                scale=alt.Scale(domain=list(MATCH_COLORS), range=list(MATCH_COLORS.values())),
                legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
            ),
            tooltip=["pair", "match", alt.Tooltip("jsd:Q", format=".3f")],
        )
    )

    # q75 rule = the SODP threshold used in tier1b_matched_pairs.py
    q75_data = df.groupby("pair")["jsd"].quantile(0.75).reset_index().rename(columns={"jsd": "q75"})
    rule = (
        alt.Chart(q75_data)
        .mark_rule(strokeDash=[4, 3], strokeWidth=1, color=NEAR_BLACK, opacity=0.5)
        .encode(
            x="q75:Q",
            y=alt.Y("pair:N", sort=PAIR_ORDER),
        )
    )

    return save(
        (strip + rule).properties(
            title="Procedural distance by outcome agreement (matched instances)",
            width=400,
            height=180,
        ),
        "fig12_tier1b_sodp",
    )


## Figure 13: Discriminative bigrams


def fig_discriminative_bigrams():
    """Top transitions that distinguish each agent pair.

    For the two most contrastive pairs, shows the bigrams (consecutive
    atom pairs) most over-represented in each agent. The length of each
    bar is the probability excess over the other agent.
    """
    src = RES / "discriminative_bigrams_v1.json"
    if not src.exists():
        print("    discriminative_bigrams_v1.json missing — skipping fig13")
        return None

    data = json.loads(src.read_text())["discriminative_bigrams"]

    FOCUS_PAIRS = [
        "Claude-4 Sonnet||GPT-4",
        "Claude-4 Sonnet||SWE-agent-LM-32B",
    ]
    PAIR_LABELS = {
        "Claude-4 Sonnet||GPT-4": "Claude-4 vs GPT-4",
        "Claude-4 Sonnet||SWE-agent-LM-32B": "Claude-4 vs SWE-LM-32B",
    }
    AGENT_COLORS_LOCAL = {
        "Claude-4 Sonnet": "#5DADE2",
        "GPT-4": "#7B241C",
        "SWE-agent-LM-32B": "#1A7A4A",
    }

    rows = []
    for pair_key in FOCUS_PAIRS:
        if pair_key not in data:
            continue
        pair_label = PAIR_LABELS[pair_key]
        agent_a, agent_b = pair_key.split("||")
        for side, agent_name in [(agent_a, agent_a), (agent_b, agent_b)]:
            key = f"{side}_signature"
            for entry in data[pair_key].get(key, [])[:5]:
                bg = entry["bigram"]
                delta = abs(entry["delta_p"])
                src_a, tgt = bg.split("|")
                label = f"{src_a} → {tgt}"
                rows.append(
                    {
                        "pair": pair_label,
                        "agent": agent_name,
                        "bigram": label,
                        "delta_p": delta,
                    }
                )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    agent_names = list(AGENT_COLORS_LOCAL.keys())

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("delta_p:Q", title="Probability excess (Δp)", axis=alt.Axis(format=".2f")),
            y=alt.Y(
                "bigram:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelFontSize=9)
            ),
            color=alt.Color(
                "agent:N",
                scale=alt.Scale(
                    domain=agent_names, range=[AGENT_COLORS_LOCAL[a] for a in agent_names]
                ),
                legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
            ),
            facet=alt.Facet(
                "pair:N", columns=1, header=alt.Header(titleFontSize=10, labelFontSize=10)
            ),
            tooltip=["pair", "agent", "bigram", alt.Tooltip("delta_p:Q", format=".4f")],
        )
        .properties(width=360, height=130)
    )

    return save(chart, "fig13_discriminative_bigrams")


## Run all

if __name__ == "__main__":
    print("Generating figures → results/paper_figures/")
    print()

    fns = [
        ("fig01 JSD heatmaps", fig_jsd_heatmaps),
        ("fig02 canonical atoms", fig_canonical_atoms),
        ("fig03 native heatstrip", fig_native_heatstrip),
        ("fig04 procgrep metrics", fig_procgrep_metrics),
        ("fig05 identification probe F1", fig_probe_f1),
        ("fig06 tokens and cost", fig_tokens_cost),
        ("fig07 file consumption", fig_files),
        ("fig08 shell taxonomy", fig_shell_taxonomy),
        ("fig09 OOD scores", fig_ood),
        ("fig10 within-trajectory drift", fig_drift),
        ("fig11a discriminative agents", fig_discriminative_agents),
        ("fig11b discriminative outcomes", fig_discriminative_outcomes),
        ("fig12 Tier 1B SODP scatter", fig_tier1b),
        ("fig13 discriminative bigrams", fig_discriminative_bigrams),
    ]

    paths = []
    for label, fn in fns:
        print(f"  {label}...")
        try:
            p = fn()
            if p:
                paths.append(p)
        except Exception as e:
            print(f"    ERROR: {e}")

    print()
    print(f"Done — {len(paths)}/{len(fns)} figures saved to results/paper_figures/")
    for p in paths:
        print(f"  {p.name}")


## Figure 14: Positional divergence line chart


def fig_positional_divergence():
    """Mean pairwise JSD by absolute step position across all agent pairs.

    Shows that step 0 (always think) has zero divergence, step 1 (opening
    action) has maximum divergence, then a slow increase through mid-trajectory.
    """
    src = RES / "positional_divergence_v1.json"
    if not src.exists():
        print("    positional_divergence_v1.json missing — skipping fig14")
        return None

    d = json.loads(src.read_text())
    abs_data = d.get("absolute", {})

    rows = [{"step": int(k), "jsd": v} for k, v in abs_data.items()]
    df = pd.DataFrame(rows).sort_values("step")

    line = (
        alt.Chart(df)
        .mark_line(color="#2471A3", strokeWidth=2)
        .encode(
            x=alt.X("step:Q", title="Step position (absolute)", scale=alt.Scale(domain=[0, 39])),
            y=alt.Y("jsd:Q", title="Mean pairwise JSD", scale=alt.Scale(domain=[0, 0.3])),
            tooltip=["step", alt.Tooltip("jsd:Q", format=".3f")],
        )
    )
    points = (
        alt.Chart(df)
        .mark_circle(color="#2471A3", size=30)
        .encode(
            x="step:Q",
            y="jsd:Q",
        )
    )
    peak = df.loc[df["jsd"].idxmax()]
    ann = (
        alt.Chart(
            pd.DataFrame(
                [
                    {
                        "step": int(peak.step),
                        "jsd": float(peak.jsd),
                        "label": f"step {int(peak.step)}: JSD={peak.jsd:.2f}",
                    }
                ]
            )
        )
        .mark_text(align="left", dx=6, dy=-8, fontSize=9, color="#2471A3")
        .encode(x="step:Q", y="jsd:Q", text="label:N")
    )
    return save(
        (line + points + ann).properties(
            title="Procedural divergence by trajectory position",
            width=380,
            height=200,
        ),
        "fig14_positional_divergence",
    )
