"""End-to-end demonstration of the gumtree adapter.

The four built-in adapters (SWE-agent, Agentless, DARS, Moatless)
target *agent-action* traces — each step is a tool call like "edit"
or "run_test". The gumtree adapter targets *AST edit scripts* instead:
each step is one node-level edit (insert / delete / update / move),
emitted at fine granularity (``ast_insert:MethodInvocation``,
``ast_delete:Identifier``, ...). This lets procgrep operate on the
output of the gumtree CLI (https://github.com/GumTreeDiff/gumtree),
which is language-neutral at the operation layer.

This script:

1. Loads the bundled multi-language fixture
   ``examples/synthetic_gumtree_traces.jsonl`` (Python, JavaScript,
   Java edits across two agents).
2. Canonicalizes via the gumtree adapter — node-typed atoms.
3. Reports the vocabulary that gumtree atoms induce.
4. Fits a BPE procedure vocabulary and prints the per-agent JSD.
5. Demonstrates the optional ``parse_gumtree_jsondiff`` helper on a
   hand-crafted raw gumtree JSON payload, showing the converted
   ``actions`` shape that the adapter consumes.

Run from the repository root:

    python examples/python/12_gumtree_adapter.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from procgrep import (
    canonicalize,
    encode,
    fit_bpe,
    jsd_matrix,
    parse_gumtree_jsondiff,
)
from procgrep.io import read_jsonl

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
        "--vocab-size",
        type=int,
        default=50,
        help="BPE target vocabulary size (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = list(read_jsonl(args.traces))
    traces = canonicalize(raw, adapter="gumtree")

    print(f"loaded {len(traces)} traces via the gumtree adapter")
    print(f"  agents:    {sorted({t.agent for t in traces})}")
    print(f"  languages: {sorted({t.group or '?' for t in traces})}")

    # Atom-level vocabulary
    atom_counter: Counter[str] = Counter()
    for t in traces:
        atom_counter.update(t.atoms)
    print(f"\nfine-grained atom vocabulary: {len(atom_counter)} distinct atoms")
    print("top 10 atoms by raw count:")
    for atom, count in atom_counter.most_common(10):
        print(f"  {count:>4d}  {atom}")

    # BPE + per-agent JSD
    vocab = fit_bpe((t.atoms for t in traces), vocab_size=args.vocab_size, seed=0)
    print(f"\nfit a BPE vocabulary of size {vocab.size}")

    fingerprints = encode(traces, vocab=vocab)
    matrix = jsd_matrix(fingerprints, group_by="agent")
    print("\npairwise JSD by agent (BPE procedure fingerprints):")
    for r in matrix.to_records():
        if str(r["row"]) < str(r["col"]):
            print(f"  {r['row']:15s} vs {r['col']:15s}  JSD = {r['jsd']:.4f}")

    # Raw gumtree JSON -> adapter input
    # Illustrate the optional `parse_gumtree_jsondiff` helper. The payload
    # below mimics what `gumtree jsondiff <before.py> <after.py>` would emit
    # for a small refactor (insert a method call, delete an identifier).
    raw_gumtree_payload = {
        "matches": [],
        "actions": [
            {
                "action": "insert-tree",
                "tree": "Call [10,30]",
                "parent": "FunctionDef [0,40]",
                "at": 2,
            },
            {"action": "delete-node", "tree": "Name: foo [12,15]"},
            {"action": "update-node", "tree": "Constant: 'old' [20,25]", "label": "'new'"},
        ],
    }
    parsed_actions = parse_gumtree_jsondiff(raw_gumtree_payload)
    print("\nparse_gumtree_jsondiff converted the raw payload into:")
    for entry in parsed_actions:
        print(f"  {entry}")

    # Build a one-trace procgrep record from the parsed actions and run it
    # through the same adapter.
    record = {
        "trace_id": "demo-raw-gumtree",
        "agent": "demo-agent",
        "group": "python",
        "actions": parsed_actions,
    }
    [demo_trace] = canonicalize([record], adapter="gumtree")
    print(f"\ncanonicalized atoms from the raw payload: {demo_trace.atoms}")


if __name__ == "__main__":
    main()
