"""Cascade figures for the interaction-surface work: how development bursts and
file co-edits are distributed by interface.

Two figures, both log-log complementary-CDFs (the distributions are heavy-tailed,
so a CCDF on log axes is the honest view):

  cascade_size_ccdf.png  -- cascade size = agent actions per human prompt, by
                            interface/mode (Claude Code, Cursor agent, Cursor chat).
  file_degree_ccdf.png   -- file co-edit degree = how many other files a file is
                            edited alongside within a cascade, by interface.

Data is the author's own local Claude Code transcripts and the live Cursor
``state.vscdb``; only aggregate distribution counts reach the figure (no paths,
no content), so regenerate on the author's machine. n=1 developer -- preliminary.
"""

from __future__ import annotations

import glob
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from procgrep.ingest.adapters.cursor_vscdb import _EDIT_TOOLS, _atom_for_bubble, _normalize_tool

# Project palette (theme.css).
COPPER, BLUE, OLIVE, RULE, INK = "#CB4D20", "#5692E5", "#585E53", "#d9d4cc", "#14110E"
DROP = {"think", "other"}  # action atoms only
CC_EDIT = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
LIVE = str(Path("~/Library/Application Support/Cursor/User/globalStorage/state.vscdb").expanduser())


def _cc_turns_and_edits():
    """Per-cascade (between human prompt) action counts + edited-file lists for Claude Code."""
    sizes, edit_cascades = [], []
    files = sorted(glob.glob(str(Path("~/.claude/projects").expanduser() / "*" / "*.jsonl")))[-40:]
    for f in files:
        seq, efiles = [], []
        for line in Path(f).read_text(errors="ignore").splitlines():
            if not line.strip().startswith("{"):
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = e.get("message")
            if not isinstance(msg, dict):
                continue
            cont = msg.get("content")
            # A genuine human prompt (not a tool-result, which is also role=user) closes a cascade.
            if msg.get("role") == "user" and (
                isinstance(cont, str)
                or (isinstance(cont, list)
                    and any(isinstance(c, dict) and c.get("type") == "text" for c in cont)
                    and not any(isinstance(c, dict) and c.get("type") == "tool_result" for c in cont))
            ):
                if seq:
                    sizes.append(len(seq))
                edit_cascades.append(efiles)
                seq, efiles = [], []
            if isinstance(cont, list):
                for c in cont:
                    if not isinstance(c, dict) or c.get("type") != "tool_use":
                        continue
                    name = c.get("name")
                    if name in CC_EDIT:
                        fp = (c.get("input") or {}).get("file_path")
                        if fp:
                            efiles.append(fp)
                        seq.append("edit")
                    elif name:
                        seq.append("act")
        if seq:
            sizes.append(len(seq))
        edit_cascades.append(efiles)
    return [s for s in sizes if s > 0], edit_cascades


def _cursor_turns_and_edits():
    """Same, per Cursor unifiedMode, from the live state.vscdb."""
    con = sqlite3.connect(f"file:{LIVE}?mode=ro&immutable=1", uri=True)
    comps = []
    for k, v in con.cursor().execute(
        "SELECT key, value FROM cursorDiskKV WHERE key >= 'composerData:' AND key < 'composerData;'"
    ).fetchall():
        try:
            o = json.loads(v)
        except json.JSONDecodeError:
            continue
        comps.append((k.split(":", 1)[1], o.get("unifiedMode") or "?",
                      [h.get("bubbleId") for h in o.get("fullConversationHeadersOnly", []) if h.get("bubbleId")]))
    c2 = con.cursor()
    sizes = defaultdict(list)
    edit_cascades = defaultdict(list)
    for cid, mode, hdrs in comps:
        seq, efiles = [], []
        for bid in hdrs:
            r = c2.execute("SELECT value FROM cursorDiskKV WHERE key=?", (f"bubbleId:{cid}:{bid}",)).fetchone()
            if not r:
                continue
            try:
                b = json.loads(r[0])
            except json.JSONDecodeError:
                continue
            if b.get("type") == 1:
                if seq:
                    sizes[mode].append(len(seq))
                edit_cascades[mode].append(efiles)
                seq, efiles = [], []
            else:
                atom = _atom_for_bubble(b)
                if atom not in DROP:
                    seq.append(atom)
                tf = b.get("toolFormerData") or {}
                t = tf.get("name") or tf.get("tool")
                if t and _normalize_tool(t) in _EDIT_TOOLS:
                    raw = tf.get("params") or tf.get("rawArgs") or tf.get("args")
                    try:
                        pj = json.loads(raw) if isinstance(raw, str) else raw
                    except (json.JSONDecodeError, TypeError):
                        pj = None
                    if isinstance(pj, dict):
                        p = pj.get("relativeWorkspacePath") or pj.get("target_file") or pj.get("path")
                        if p:
                            efiles.append(p)
        if seq:
            sizes[mode].append(len(seq))
        edit_cascades[mode].append(efiles)
    con.close()
    return sizes, edit_cascades


def _degrees(edit_cascades):
    """File co-edit degree: within each cascade, every distinct pair of edited files is a link."""
    deg = Counter()
    for c in edit_cascades:
        files = list(dict.fromkeys(c))
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                deg[files[i]] += 1
                deg[files[j]] += 1
    return list(deg.values())


def _ccdf(xs):
    xs = np.sort(np.array([x for x in xs if x > 0]))
    n = len(xs)
    # P(X >= x) at each distinct value
    vals, idx = np.unique(xs, return_index=True)
    surv = 1.0 - idx / n
    return vals, surv


def _panel(ax, series, xlabel, title):
    for label, color, xs in series:
        if len(xs) < 20:
            continue
        v, s = _ccdf(xs)
        ax.step(v, s, where="post", color=color, lw=1.8, label=f"{label}  n={len(xs)}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel("fraction ≥ x", fontsize=10, color=INK)
    ax.set_title(title, fontsize=11, color=INK, pad=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=OLIVE, labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")


def main() -> None:
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    cc_sizes, cc_edits = _cc_turns_and_edits()
    cur_sizes, cur_edits = _cursor_turns_and_edits()
    plt.rcParams["font.family"] = "monospace"

    # Figure 1: cascade size CCDF
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _panel(ax, [
        ("Claude Code", COPPER, cc_sizes),
        ("Cursor agent", BLUE, cur_sizes.get("agent", [])),
        ("Cursor chat", OLIVE, cur_sizes.get("chat", [])),
    ], "cascade size, actions per prompt", "Cascade-size distribution by interface")
    fig.tight_layout()
    fig.savefig(out / "cascade_size_ccdf.png", dpi=200, facecolor="white")
    plt.close(fig)

    # Figure 2: file co-edit degree CCDF
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _panel(ax, [
        ("Claude Code", COPPER, _degrees(cc_edits)),
        ("Cursor agent", BLUE, _degrees(cur_edits.get("agent", []))),
        ("Cursor chat", OLIVE, _degrees(cur_edits.get("chat", []))),
    ], "file co-edit degree", "File co-edit degree by interface")
    fig.tight_layout()
    fig.savefig(out / "file_degree_ccdf.png", dpi=200, facecolor="white")
    plt.close(fig)

    summary = {
        "cascade_size": {"Claude Code": len(cc_sizes), "Cursor agent": len(cur_sizes.get("agent", [])),
                         "Cursor chat": len(cur_sizes.get("chat", []))},
    }
    print("wrote docs/figures/cascade_size_ccdf.png + file_degree_ccdf.png")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
