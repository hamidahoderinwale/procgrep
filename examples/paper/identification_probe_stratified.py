"""Stratified k-fold identification probe: answers the *within-corpus* question:
"Given a held-out trace from one of N agents we've already seen, can a probe
identify which agent produced it?"

Distinct from procgrep's built-in ``leave_one_group_out`` (which holds out an
entire agent and is structurally 0% by construction; a probe can't predict a
label that doesn't appear in training).

Outputs:
- Per-agent precision/recall/F1 + macro-F1 + overall accuracy
- Confusion matrix (which agents get mistaken for which)
- Plots: confusion matrix + per-agent F1 bars
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

_PROCGREP_SRC = Path(__file__).resolve().parent.parent / "procgrep" / "src"
if _PROCGREP_SRC.exists() and str(_PROCGREP_SRC) not in sys.path:
    sys.path.insert(0, str(_PROCGREP_SRC))

from procgrep.bpe import fit_bpe  # noqa: E402
from procgrep.encode import encode  # noqa: E402
from procgrep.types import Trace  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT_DIR = RESULTS / "identification_probe_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_REGISTRY = [
    ("Claude-3 Opus", "fingerprints_claude3opus.jsonl"),
    ("Claude-3.5 Sonnet", "fingerprints_claude3.5sonnet.jsonl"),
    ("Claude-4 Sonnet", "fingerprints_claude4sonnet.jsonl"),
    ("SWE-agent-LM-32B", "fingerprints_child_n500.jsonl"),
    ("GPT-4", "fingerprints_gpt4.jsonl"),
    ("GPT-4o", "fingerprints_gpt4o.jsonl"),
    # Newer entries (will be loaded if they exist after the Track-B pull lands)
    ("DARS+DeepSeek-R1", "fingerprints_dars_r1.jsonl"),
    ("OpenHands+Claude-4", "fingerprints_openhands_claude4.jsonl"),
    ("Claude-4 Opus (tools)", "fingerprints_claude4opus_tools.jsonl"),
]


def load_all(layer: str = "canonical") -> tuple[list[Trace], list[str]]:
    traces: list[Trace] = []
    labels: list[str] = []
    for name, fname in AGENT_REGISTRY:
        path = RESULTS / fname
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            d = json.loads(line)
            atoms = d.get(f"atoms_{layer}", [])
            if not atoms:
                continue
            traces.append(Trace(trace_id=d["instance_id"], agent=name, atoms=atoms))
            labels.append(name)
    return traces, labels


def run_probe(layer: str, vocab_size: int) -> dict:
    """Fit BPE, encode trajectories, run stratified 5-fold logistic regression."""
    traces, labels = load_all(layer)
    print(f"[{layer}] loaded {len(traces)} traces across {len(set(labels))} agents")

    # Down-sample SWE-agent-LM-32B to match the others' n=50 so the probe
    # isn't dominated by class imbalance.
    agents_n = Counter(labels)
    target_n = min(agents_n.values())
    if max(agents_n.values()) > 4 * target_n:
        # Subsample within each agent's contiguous block
        kept_traces = []
        kept_labels = []
        per_agent_kept = Counter()
        for t, lab in zip(traces, labels, strict=False):
            if per_agent_kept[lab] < target_n:
                kept_traces.append(t)
                kept_labels.append(lab)
                per_agent_kept[lab] += 1
        traces, labels = kept_traces, kept_labels
        print(f"[{layer}] after balancing: {len(traces)} traces, " f"{dict(Counter(labels))}")

    sequences = [t.atoms for t in traces]
    vocab = fit_bpe(sequences, vocab_size=vocab_size, seed=0)
    fps = encode(traces, vocab=vocab)
    # fp.distribution() is L1-normalized over the BPE vocab
    X = np.stack([fp.distribution() for fp in fps])
    y = np.array(labels)
    classes = sorted(set(labels))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    all_y_true, all_y_pred = [], []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        all_y_true.extend(y[te])
        all_y_pred.extend(pred)
    overall_acc = float(np.mean(np.array(all_y_true) == np.array(all_y_pred)))
    macro_f1 = float(f1_score(all_y_true, all_y_pred, average="macro"))
    report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=classes)

    print(f"[{layer}] overall accuracy: {overall_acc:.3f}")
    print(f"[{layer}] random baseline:  {1.0 / len(classes):.3f}")
    print(f"[{layer}] macro F1:         {macro_f1:.3f}")
    print(f"[{layer}] per-agent F1:")
    for c in classes:
        f1 = report.get(c, {}).get("f1-score", 0)
        print(f"  {c:28s}  F1={f1:.3f}")

    fig, ax = plt.subplots(figsize=(1 + 0.7 * len(classes), 1 + 0.7 * len(classes)))
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted agent")
    ax.set_ylabel("true agent")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, label="count")
    ax.set_title(
        f"Identification probe confusion matrix ({layer}, "
        f"acc={overall_acc:.2f}, macro_F1={macro_f1:.2f}, BPE V={vocab.size})"
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"confusion_{layer}.png", dpi=150)
    plt.close()

    return {
        "layer": layer,
        "vocab_size": vocab.size,
        "n_traces": len(traces),
        "n_agents": len(classes),
        "overall_accuracy": overall_acc,
        "random_baseline": 1.0 / len(classes),
        "macro_f1": macro_f1,
        "per_agent_f1": {c: report.get(c, {}).get("f1-score", 0) for c in classes},
        "confusion_classes": classes,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    print("=" * 80)
    print("Stratified k-fold identification probe (within-corpus)")
    print("=" * 80)
    print()
    canon = run_probe("canonical", vocab_size=64)
    print()
    native = run_probe("native", vocab_size=128)

    out = {"canonical": canon, "native": native}
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_DIR}/summary.json + confusion_*.png")


if __name__ == "__main__":
    main()
