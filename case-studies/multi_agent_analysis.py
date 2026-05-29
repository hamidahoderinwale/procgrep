"""Multi-agent procedural fingerprint analysis (Tier 1A + 1B).

Joins fingerprint JSONLs across agents, then runs:

1. Per-agent canonical + native atom distributions (mass-share table)
2. Pairwise JSD matrix at canonical + native (Q-B: who's similar to whom?)
3. Leave-one-group-out identification probe (Q-A: how identifiable are agents?)
4. Discriminative procedures between the case-study distillation pair
5. Plots: distribution heatmap, JSD matrix, identification confusion matrix

All $0 inference. Operates on the JSONL files produced by pull_and_fingerprint.py.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROCGREP_SRC = Path(__file__).resolve().parent.parent / "procgrep" / "src"
if _PROCGREP_SRC.exists() and str(_PROCGREP_SRC) not in sys.path:
    sys.path.insert(0, str(_PROCGREP_SRC))

from procgrep import jsd_matrix  # noqa: E402
from procgrep.bpe import fit_bpe  # noqa: E402
from procgrep.encode import encode  # noqa: E402
from procgrep.probe import leave_one_group_out  # noqa: E402
from procgrep.types import Trace  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT_DIR = RESULTS / "multi_agent_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (display_name, fingerprint_file, family, scaffold)
AGENT_REGISTRY = [
    # SWE-agent format (same scaffold, best comparability)
    ("Claude-3 Opus", "fingerprints_claude3opus.jsonl", "Anthropic", "SWE-agent"),
    ("Claude-3.5 Sonnet", "fingerprints_claude3.5sonnet.jsonl", "Anthropic", "SWE-agent"),
    ("Claude-4 Sonnet", "fingerprints_claude4sonnet.jsonl", "Anthropic", "SWE-agent"),
    ("SWE-agent-LM-32B", "fingerprints_child_n500.jsonl", "Qwen+SFT", "SWE-agent"),
    ("GPT-4", "fingerprints_gpt4.jsonl", "OpenAI", "SWE-agent"),
    ("GPT-4o", "fingerprints_gpt4o.jsonl", "OpenAI", "SWE-agent"),
    # OpenHands format (different scaffold; included for model-family coverage)
    ("GPT-5", "fingerprints_gpt5.jsonl", "OpenAI", "OpenHands"),
    ("Claude-4.5 Sonnet", "fingerprints_claude4.5sonnet.jsonl", "Anthropic", "OpenHands"),
]


def load_traces(path: Path, agent_name: str, layer: str) -> list[Trace]:
    if not path.exists():
        return []
    traces: list[Trace] = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        traces.append(
            Trace(
                trace_id=d["instance_id"],
                agent=agent_name,
                atoms=d[f"atoms_{layer}"],
                metadata={"resolved": d.get("resolved"), "n_steps": d.get("n_steps")},
            )
        )
    return traces


def main() -> None:
    # 1. Load all agents' traces at canonical + native
    canonical_traces: list[Trace] = []
    native_traces: list[Trace] = []
    summary: list[dict] = []
    for name, fname, family, scaffold in AGENT_REGISTRY:
        path = RESULTS / fname
        c = load_traces(path, name, "canonical")
        n = load_traces(path, name, "native")
        canonical_traces.extend(c)
        native_traces.extend(n)
        n_act = sum(len([a for a in t.atoms if a != "think"]) for t in c)
        summary.append(
            {
                "agent": name,
                "family": family,
                "scaffold": scaffold,
                "n_traces": len(c),
                "n_action_atoms": n_act,
            }
        )
        print(f"  {name:24s} traces={len(c):4d}  action_atoms={n_act:6d}  ({family} / {scaffold})")
    print()

    # 2. Per-agent canonical atom distribution
    by_agent_canon: dict[str, Counter] = {}
    for t in canonical_traces:
        c = by_agent_canon.setdefault(t.agent, Counter())
        for a in t.atoms:
            if a != "think":
                c[a] += 1
    all_canon_atoms = sorted({a for c in by_agent_canon.values() for a in c})
    print("=" * 88)
    print("Per-agent canonical atom mass (%)")
    print("=" * 88)
    header = f"{'agent':24s}  " + "  ".join(f"{a[:8]:>8s}" for a in all_canon_atoms)
    print(header)
    for name, fname, *_ in AGENT_REGISTRY:
        c = by_agent_canon.get(name, Counter())
        total = sum(c.values())
        if total == 0:
            continue
        row = "  ".join(f"{100 * c.get(a, 0) / total:>7.1f}%" for a in all_canon_atoms)
        print(f"{name:24s}  {row}")
    print()

    # 3. Distribution heatmap
    agent_names = [
        name
        for name, fname, *_ in AGENT_REGISTRY
        if (RESULTS / fname).exists() and sum(by_agent_canon.get(name, Counter()).values()) > 0
    ]
    matrix = np.array(
        [
            [
                by_agent_canon[a].get(atom, 0) / max(1, sum(by_agent_canon[a].values()))
                for atom in all_canon_atoms
            ]
            for a in agent_names
        ]
    )
    fig, ax = plt.subplots(figsize=(11, 1 + 0.6 * len(agent_names)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(all_canon_atoms)))
    ax.set_xticklabels(all_canon_atoms, rotation=30, ha="right")
    ax.set_yticks(range(len(agent_names)))
    ax.set_yticklabels(agent_names)
    plt.colorbar(im, ax=ax, label="atom mass share")
    ax.set_title(f"Canonical procedural fingerprints across {len(agent_names)} agents")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "canonical_heatmap.png", dpi=150)
    plt.close()
    print("Saved: canonical_heatmap.png")

    # 4. JSD matrices at canonical and native via procgrep.jsd_matrix
    # We need to fit a BPE vocabulary first so the encode step has a vocab.
    print()
    print("=" * 88)
    print("Fitting BPE + encoding for JSD matrix (canonical)")
    print("=" * 88)
    canonical_seqs = [t.atoms for t in canonical_traces if t.atoms]
    vocab_c = fit_bpe(canonical_seqs, vocab_size=64, seed=0)
    fps_c = encode(canonical_traces, vocab=vocab_c)
    print(f"  canonical vocab size: {vocab_c.size}, n_fingerprints={len(fps_c)}")

    native_seqs = [t.atoms for t in native_traces if t.atoms]
    vocab_n = fit_bpe(native_seqs, vocab_size=128, seed=0)
    fps_n = encode(native_traces, vocab=vocab_n)
    print(f"  native vocab size:    {vocab_n.size}, n_fingerprints={len(fps_n)}")

    mat_c = jsd_matrix(fps_c, group_by="agent")
    mat_n = jsd_matrix(fps_n, group_by="agent")

    # Render the matrix as a heatmap. JsdMatrix exposes ``to_array()`` (numpy
    # 2D array) and ``to_records()`` (list of {row, col, jsd} dicts); we extract
    # group names from the first ``row`` value per unique-row in records.
    for label, mat, suffix in [("canonical", mat_c, "canonical"), ("native", mat_n, "native")]:
        values = mat.to_array()
        records = mat.to_records()
        # Group names in canonical order (preserve first-seen order in records)
        seen: list[str] = []
        for r in records:
            if r["row"] not in seen:
                seen.append(r["row"])
        groups = seen
        fig, ax = plt.subplots(figsize=(1 + 0.7 * len(groups), 1 + 0.7 * len(groups)))
        im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=float(values.max()))
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=30, ha="right")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups)
        for i in range(len(groups)):
            for j in range(len(groups)):
                ax.text(
                    j,
                    i,
                    f"{values[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if values[i][j] < float(values.max()) / 2 else "black",
                    fontsize=8,
                )
        plt.colorbar(im, ax=ax, label="JSD")
        ax.set_title(
            f"Pairwise JSD between agents ({label} alphabet, BPE V={vocab_c.size if label == 'canonical' else vocab_n.size})"
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"jsd_matrix_{suffix}.png", dpi=150)
        plt.close()
        print(f"Saved: jsd_matrix_{suffix}.png")

    # 5. Identification probe
    print()
    print("=" * 88)
    print("Identification probe (canonical, leave-one-group-out by agent)")
    print("=" * 88)
    res_c = leave_one_group_out(fps_c, label_field="agent")
    print(f"  overall accuracy:   {res_c.overall_accuracy:.3f}")
    print(f"  baseline (random):  {1.0 / len(agent_names):.3f}")
    print("  per-agent accuracy:")
    for name in agent_names:
        acc = res_c.per_group_accuracy.get(name)
        print(f"    {name:28s}  {acc:.3f}" if acc is not None else f"    {name:28s}  (no data)")

    print()
    print("=" * 88)
    print("Identification probe (native)")
    print("=" * 88)
    res_n = leave_one_group_out(fps_n, label_field="agent")
    print(f"  overall accuracy:   {res_n.overall_accuracy:.3f}")
    print("  per-agent accuracy:")
    for name in agent_names:
        acc = res_n.per_group_accuracy.get(name)
        print(f"    {name:28s}  {acc:.3f}" if acc is not None else f"    {name:28s}  (no data)")

    # 6. Summary JSON
    out = {
        "agents": summary,
        "jsd_canonical": {"records": mat_c.to_records()},
        "jsd_native": {"records": mat_n.to_records()},
        "id_probe_canonical": {
            "overall_accuracy": float(res_c.overall_accuracy),
            "per_group_accuracy": dict(res_c.per_group_accuracy),
        },
        "id_probe_native": {
            "overall_accuracy": float(res_n.overall_accuracy),
            "per_group_accuracy": dict(res_n.per_group_accuracy),
        },
        "bpe_vocab_sizes": {"canonical": vocab_c.size, "native": vocab_n.size},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    print("\nSaved: summary.json")


if __name__ == "__main__":
    main()
