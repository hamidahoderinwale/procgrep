"""Agent attribution from procedural fingerprints, with three naive baselines.

The question this script answers:

    Given a trace produced under some held-out condition, can we
    correctly identify the agent that produced it from its
    procedural fingerprint alone? And does the BPE motif vocabulary
    earn its keep relative to dumber representations of the same
    trace?

The setup is the leave-one-group-out probe (`procgrep.probe`) with
``label_field="agent"``. Each held-out group's traces are predicted
by a classifier trained on the other groups. If procedural style is
agent-bound and language-/task-portable, attribution accuracy stays
high across the LOGO folds.

We compare the BPE-motif fingerprint against three naive baselines:

1. **Raw atom-frequency.** Skip BPE entirely; encode each trajectory
   as its atom-level L1-normalized distribution. Tests whether BPE
   motifs carry information beyond the marginal atom distribution.
2. **Length-vector.** Each trajectory becomes a 1-D feature: its
   atom count. Tests whether attribution is reducible to "this agent
   produces longer traces."
3. **Majority class.** Predict the most-frequent agent label in the
   training fold. Floor.

The script runs on the bundled multi-language gumtree fixture
(``examples/synthetic_gumtree_traces.jsonl``), where the agents have
deliberately different procedural signatures (``agent_alpha`` =
surgical updates, ``agent_beta`` = rip-and-replace). Pass ``--traces``
to point at a different corpus.

Run from the repository root:

    python examples/python/10_agent_attribution.py
    python examples/python/10_agent_attribution.py --traces my.jsonl
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from procgrep import (
    Fingerprint,
    Trace,
    canonicalize,
    encode,
    fit_bpe,
    leave_one_group_out,
)
from procgrep.io import read_jsonl

# sklearn >= 1.8 emits a FutureWarning for `penalty=`; the keyword is still
# valid and used throughout `procgrep.probe`. Silence the demo script's
# output so the table is readable.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = ROOT / "examples" / "synthetic_gumtree_traces.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES,
        help="JSONL trace file (default: bundled gumtree multi-language fixture)",
    )
    parser.add_argument(
        "--adapter",
        default="gumtree",
        help="canonicalize adapter (default: gumtree)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50,
        help="BPE target vocabulary size (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for the classifier (default: 0)",
    )
    return parser.parse_args()


def atom_frequency_distribution(trace: Trace, vocab: list[str]) -> np.ndarray:
    """L1-normalized atom-frequency vector over the union of atoms in the corpus."""
    counts = Counter(trace.atoms)
    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(vocab))
    return np.array([counts.get(a, 0) / total for a in vocab])


def evaluate_logo(
    x: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[float, dict[str, float]]:
    """Run LOGO with a logistic regression and return overall + per-group accuracy."""
    per_group: dict[str, float] = {}
    accuracies: list[float] = []
    for train_idx, test_idx in LeaveOneGroupOut().split(x, labels, groups):
        if len(set(labels[train_idx])) < 2:
            # Classifier requires at least two classes in the training fold.
            continue
        held_out = groups[test_idx[0]]
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=seed
        )
        clf.fit(x[train_idx], labels[train_idx])
        predictions = clf.predict(x[test_idx])
        acc = float(np.mean(predictions == labels[test_idx]))
        per_group[str(held_out)] = acc
        accuracies.append(acc)
    overall = float(np.mean(accuracies)) if accuracies else 0.0
    return overall, per_group


def majority_class_logo(labels: np.ndarray, groups: np.ndarray) -> tuple[float, dict[str, float]]:
    """LOGO accuracy when the predictor always returns the training-fold majority class."""
    per_group: dict[str, float] = {}
    accuracies: list[float] = []
    for train_idx, test_idx in LeaveOneGroupOut().split(labels.reshape(-1, 1), labels, groups):
        train_labels = labels[train_idx]
        if len(train_labels) == 0:
            continue
        majority = Counter(train_labels).most_common(1)[0][0]
        held_out = groups[test_idx[0]]
        predictions = np.array([majority] * len(test_idx))
        acc = float(np.mean(predictions == labels[test_idx]))
        per_group[str(held_out)] = acc
        accuracies.append(acc)
    overall = float(np.mean(accuracies)) if accuracies else 0.0
    return overall, per_group


def main() -> None:
    args = parse_args()
    raw = list(read_jsonl(args.traces))
    traces = canonicalize(raw, adapter=args.adapter)

    # The LOGO probe needs a `group` label per trace. The gumtree
    # fixture uses language as the group; you can swap to any
    # metadata field by editing the fixture or pre-grouping yourself.
    if any(t.group is None for t in traces):
        raise SystemExit("corpus has traces without a 'group' label; LOGO attribution needs one.")

    agents = sorted({t.agent for t in traces})
    groups = sorted({t.group or "" for t in traces})
    print(f"loaded {len(traces)} traces; {len(agents)} agents x {len(groups)} groups")
    print(f"  agents : {agents}")
    print(f"  groups : {groups}")

    # --- Representation 1: BPE motif fingerprint ----------------------------
    vocab = fit_bpe((t.atoms for t in traces), vocab_size=args.vocab_size, seed=args.seed)
    fps: list[Fingerprint] = encode(traces, vocab=vocab)
    bpe_result = leave_one_group_out(fps, label_field="agent", seed=args.seed)

    # --- Representation 2: raw atom-frequency -------------------------------
    atom_vocab = sorted({a for t in traces for a in t.atoms})
    x_atoms = np.stack([atom_frequency_distribution(t, atom_vocab) for t in traces], axis=0)
    labels = np.array([t.agent for t in traces])
    group_arr = np.array([t.group for t in traces])
    atom_overall, atom_per = evaluate_logo(x_atoms, labels, group_arr, seed=args.seed)

    # --- Representation 3: length-only --------------------------------------
    x_len = np.array([[len(t.atoms)] for t in traces], dtype=float)
    len_overall, len_per = evaluate_logo(x_len, labels, group_arr, seed=args.seed)

    # --- Floor: majority-class predictor ------------------------------------
    maj_overall, maj_per = majority_class_logo(labels, group_arr)

    # --- Report -------------------------------------------------------------
    print("\nleave-one-group-out attribution accuracy (predict agent from held-out group):")
    header = f"  {'representation':22s} {'overall':>9s}  " + "  ".join(f"{g:>10s}" for g in groups)
    print(header)
    rows = [
        ("BPE motif fingerprint", bpe_result.overall_accuracy, bpe_result.per_group_accuracy),
        ("raw atom-frequency", atom_overall, atom_per),
        ("length-only", len_overall, len_per),
        ("majority-class floor", maj_overall, maj_per),
    ]
    for name, overall, per_group in rows:
        cells = "  ".join(f"{per_group.get(g, float('nan')):>10.2f}" for g in groups)
        print(f"  {name:22s} {overall:>9.2f}  {cells}")

    print(
        "\n  interpretation: BPE-motif accuracy meaningfully above raw-atom accuracy "
        "means motifs carry information beyond marginal atom frequencies. "
        "BPE-motif near length-only or majority-class accuracy means the "
        "representation isn't earning its keep on this corpus."
    )


if __name__ == "__main__":
    main()
