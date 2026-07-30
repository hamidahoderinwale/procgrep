"""The induced procedure vocabulary as a tree: merges decomposed down to atoms.

Fits ONE pooled BPE vocabulary over every interface's atom sequences (the shared
neutral basis) and draws the deepest procedures in it. Each merge glues two
existing tokens, so every procedure is a binary tree whose leaves are atoms:
reading the leaves top-to-bottom spells the procedure. Dots at the root mark
which agents emit it.

This is the LEARNED vocabulary, not a hand-declared taxonomy: the groupings come
out of BPE over real sessions, and the procedures drawn are every one at the
maximum merge depth, so nothing is hand-picked.

Corpus: SALT-NLP/SWE-chat transcripts, 25 sessions per interface.
Style: procgrep figtheme (monospace, Tufte-minimal, no grid).
Output: docs/figures/induced_vocab_tree/induced_vocab_tree.png
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "src"))

from figtheme import BLUE, COPPER, GREEN, INK, MAGENTA, OLIVE, RULE, init, save_fig  # noqa: E402

import procgrep.ingest.adapters.claude_code as cc  # noqa: E402
import procgrep.ingest.adapters.codex as cx  # noqa: E402
import procgrep.ingest.adapters.gemini_cli as gm  # noqa: E402
import procgrep.ingest.adapters.opencode as oc  # noqa: E402
from procgrep.bpe import PROCEDURE_SEPARATOR as SEP  # noqa: E402
from procgrep.bpe import fit_bpe  # noqa: E402
from procgrep.encode import encode  # noqa: E402
from procgrep.types import Trace  # noqa: E402

REPO = "SALT-NLP/SWE-chat"
AGENTS = ["Claude Code", "OpenCode", "Codex", "Gemini CLI"]
COLORS = dict(zip(AGENTS, [BLUE, COPPER, GREEN, MAGENTA], strict=True))
VOCAB_SIZE = 48  # atoms + merges; deep enough for depth-3 procedures to appear
SESSIONS_PER_AGENT = 25
OUT = ROOT / "docs" / "figures" / "induced_vocab_tree" / "induced_vocab_tree.png"


## corpus


def _lines(path):
    """Read a JSONL transcript, skipping anything that is not an object."""
    out = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def atom_seq(agent, path):
    """Run the interface's own adapter to get its canonical atom sequence."""
    if agent == "Claude Code":
        atoms = cc.claude_code_adapter({"events": _lines(path)})
    elif agent == "OpenCode":
        atoms = oc.opencode_adapter(oc.load_opencode_session(json.loads(Path(path).read_text())))
    elif agent == "Codex":
        atoms = cx.codex_adapter(cx.load_codex_session(_lines(path)))
    else:
        text = Path(path).read_text()
        try:
            obj = json.loads(text)
        except ValueError:
            obj = _lines(path)
        atoms = gm.gemini_cli_adapter(gm.load_gemini_session(obj))
    return [str(a) for a in atoms if str(a) != "prompt_ai"]


def collect(n=SESSIONS_PER_AGENT):
    sessions = pd.read_parquet(hf_hub_download(REPO, "sessions.parquet", repo_type="dataset"))
    logs = pd.read_parquet(hf_hub_download(REPO, "session_logs.parquet", repo_type="dataset"))
    merged = sessions[["session_id", "agent"]].merge(
        logs[["session_id", "transcript_path"]], on="session_id"
    )
    merged = merged[merged.transcript_path.astype(str).str.startswith("transcripts/")]
    by = {}
    for agent in AGENTS:
        seqs = []
        for tp in list(merged[merged.agent == agent].transcript_path)[:n]:
            try:
                seq = atom_seq(agent, hf_hub_download(REPO, tp, repo_type="dataset"))
            except Exception:
                continue
            if len(seq) >= 2:
                seqs.append(seq)
        by[agent] = seqs
    return by


## vocabulary


def fit():
    by = collect()
    traces = [
        Trace(trace_id=f"{a}-{i}", agent=a, atoms=s) for a in AGENTS for i, s in enumerate(by[a])
    ]
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=VOCAB_SIZE)
    # which agents emit each token, from the same pooled encoding
    users: dict[str, set[str]] = {}
    tokens = vocab.tokens()
    for fp in encode(traces, vocab=vocab):
        for tok, count in zip(tokens, fp.counts, strict=True):
            if count:
                users.setdefault(tok, set()).add(fp.group)
    return vocab, users, len(traces)


def children_map(vocab):
    """token -> (left, right) for merges; atoms have no children."""
    return {left + SEP + right: (left, right) for left, right in vocab.merges}


def depths(vocab, kids):
    d = dict.fromkeys(vocab.atoms, 0)
    for tok, (left, right) in kids.items():
        d[tok] = 1 + max(d[left], d[right])
    return d


## layout


def layout(tok, kids, y0):
    """Place a binary merge tree: leaves stacked from y0 down, atoms at x=0.

    Returns (node, next_y). A node carries its own children rather than being
    re-matched by name later -- the two children of a merge are often the same
    token (edit+edit), so a name lookup cannot tell the subtrees apart.
    """
    if tok not in kids:
        return {"tok": tok, "x": 0.0, "y": y0, "kids": []}, y0 + 1.0
    left, right = kids[tok]
    ln, y1 = layout(left, kids, y0)
    rn, y2 = layout(right, kids, y1)
    node = {
        "tok": tok,
        "x": max(ln["x"], rn["x"]) + 1.0,
        "y": (ln["y"] + rn["y"]) / 2,
        "kids": [ln, rn],
    }
    return node, y2


def walk(node):
    yield node
    for kid in node["kids"]:
        yield from walk(kid)


def main():
    vocab, users, n_traces = fit()
    kids = children_map(vocab)
    depth = depths(vocab, kids)
    deepest = max(depth[t] for t in kids)
    shown = sorted((t for t in kids if depth[t] == deepest), key=lambda t: len(t.split(SEP)))

    init()
    roots, y = [], 0.0
    for tok in shown:
        root, y = layout(tok, kids, y)
        roots.append(root)
        y += 1.4  # gap between procedures
    ymax = y - 1.4  # drop the trailing gap
    span = max(n["x"] for r in roots for n in walk(r))

    fig, ax = plt.subplots(figsize=(6.6, max(4.0, 0.24 * ymax)))

    def ypos(v):  # first procedure at the top
        return ymax - v

    def xpos(xd):  # root at the left, atom leaves at the right
        return span - xd

    dots_x = [xpos(span) - 0.95 + i * 0.22 for i in range(len(AGENTS))]

    for root in roots:
        for node in walk(root):
            if not node["kids"]:
                ax.text(
                    xpos(0) + 0.12,
                    ypos(node["y"]),
                    node["tok"],
                    ha="left",
                    va="center",
                    fontsize=9.5,
                    color=OLIVE,
                )
                continue
            for kid in node["kids"]:
                ax.plot(
                    [xpos(node["x"]), xpos(node["x"]), xpos(kid["x"])],
                    [ypos(node["y"]), ypos(kid["y"]), ypos(kid["y"])],
                    color=RULE,
                    lw=0.9,
                    solid_joinstyle="miter",
                )
        ax.scatter(xpos(root["x"]), ypos(root["y"]), s=14, color=INK, zorder=5)
        # which agents emit this procedure
        for x, agent in zip(dots_x, AGENTS, strict=True):
            on = agent in users.get(root["tok"], ())
            ax.scatter(
                x,
                ypos(root["y"]),
                s=20,
                zorder=5,
                color=COLORS[agent] if on else "#d3cec6",
                edgecolors="none",
            )

    for x, agent in zip(dots_x, AGENTS, strict=True):
        ax.text(
            x,
            ypos(-0.5),
            agent,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=COLORS[agent],
            rotation=55,
            rotation_mode="anchor",
        )
    ax.text(
        dots_x[0],
        ypos(ymax) - 0.1,
        "filled = the agent emits this procedure",
        ha="left",
        va="center",
        fontsize=8.5,
        color=OLIVE,
    )

    ax.set_xlim(dots_x[0] - 0.2, xpos(0) + 1.35)
    ax.set_ylim(ypos(ymax) - 0.5, ypos(0) + 1.4)
    ax.axis("off")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"saved {save_fig(fig, OUT)}")
    print(
        f"  vocab {vocab.size} ({len(vocab.atoms)} atoms + {len(vocab.merges)} merges), "
        f"{n_traces} sessions, depth-{deepest} procedures: {len(shown)}"
    )
    for tok in shown:
        print(f"  {tok}  <- {sorted(users.get(tok, ()))}")
    print("  depth histogram:", dict(sorted(Counter(depth[t] for t in kids).items())))


if __name__ == "__main__":
    main()
