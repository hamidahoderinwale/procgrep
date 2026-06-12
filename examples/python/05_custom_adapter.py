"""Custom adapter: plug a new scaffold into procgrep.

The TraceAdapter protocol is a callable that takes one raw trace
record and returns an `AtomSequence`. procgrep ships built-in
adapters for SWE-agent, Agentless, DARS, and Moatless; any other
scaffold plugs in by registering a callable that knows the
scaffold's trace shape.

This example registers a minimal adapter for a fictional "Looper"
scaffold whose action names differ from the built-ins, then runs
the rest of the pipeline against the adapter's output.

Run from the repository root:

    python examples/python/05_custom_adapter.py
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from procgrep import canonicalize, encode, fit_bpe, jsd_matrix
from procgrep.canonicalize import register_adapter
from procgrep.types import (
    ATOM_EDIT,
    ATOM_LOCALIZE,
    ATOM_OTHER,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SUBMIT,
    AtomSequence,
)


def looper_adapter(record: Mapping[str, Any]) -> AtomSequence:
    """Map Looper-format traces into canonical atoms.

    Looper traces carry a list of dicts under ``steps``, each with
    an ``op`` field naming the operation. Map each operation to a
    canonical atom; unknown operations fall through to ATOM_OTHER.
    """
    op_to_atom = {
        "FIND": ATOM_LOCALIZE,
        "VIEW": ATOM_READ_FILE,
        "PATCH": ATOM_EDIT,
        "EXECUTE": ATOM_RUN_TEST,
        "FINISH": ATOM_SUBMIT,
    }
    atoms: AtomSequence = []
    steps = record.get("steps", [])
    if not isinstance(steps, list):
        return atoms
    for step in steps:
        if isinstance(step, Mapping):
            atoms.append(op_to_atom.get(str(step.get("op")), ATOM_OTHER))
    return atoms


def main() -> None:
    register_adapter("looper", looper_adapter, overwrite=True)

    raw_traces: list[dict[str, Any]] = [
        {
            "trace_id": "loop-001",
            "agent": "looper-alpha",
            "group": "control",
            "steps": [
                {"op": "FIND"},
                {"op": "VIEW"},
                {"op": "PATCH"},
                {"op": "EXECUTE"},
                {"op": "FINISH"},
            ],
        },
        {
            "trace_id": "loop-002",
            "agent": "looper-alpha",
            "group": "control",
            "steps": [
                {"op": "FIND"},
                {"op": "PATCH"},
                {"op": "PATCH"},
                {"op": "PATCH"},
                {"op": "EXECUTE"},
                {"op": "FINISH"},
            ],
        },
        {
            "trace_id": "loop-003",
            "agent": "looper-beta",
            "group": "control",
            "steps": [
                {"op": "VIEW"},
                {"op": "VIEW"},
                {"op": "PATCH"},
                {"op": "FINISH"},
            ],
        },
    ]

    traces = canonicalize(raw_traces, adapter="looper")
    print("canonicalized Looper traces:")
    for t in traces:
        print(f"  {t.trace_id:10s} agent={t.agent:15s} atoms={t.atoms}")

    vocab = fit_bpe((t.atoms for t in traces), vocab_size=10, seed=0)
    fingerprints = encode(traces, vocab=vocab)
    matrix = jsd_matrix(fingerprints, group_by="agent")

    print("\nJSD by agent on the Looper corpus:")
    for r in matrix.to_records():
        if str(r["row"]) < str(r["col"]):
            print(f"  {r['row']:15s} vs {r['col']:15s}  JSD = {r['jsd']:.4f}")


if __name__ == "__main__":
    main()
