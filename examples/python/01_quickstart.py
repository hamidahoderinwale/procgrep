"""End-to-end procgrep pipeline on the bundled synthetic corpus.

Demonstrates the canonical workflow expressed through the Python API:

    raw JSONL -> canonicalize -> fit_bpe -> encode -> jsd_matrix

Mirrors the CLI quickstart in `examples/README.md`. Run from the
repository root:

    python examples/python/01_quickstart.py
"""

from __future__ import annotations

from pathlib import Path

from procgrep import canonicalize, encode, fit_bpe, jsd_matrix
from procgrep.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "examples" / "synthetic_traces.jsonl"


def main() -> None:
    raw = list(read_jsonl(TRACES))
    print(f"loaded {len(raw)} raw records from {TRACES.name}")

    traces = canonicalize(raw, adapter="swe-agent")
    print(f"canonicalized into {len(traces)} traces")
    for trace in traces:
        print(f"  {trace.trace_id:10s} agent={trace.agent:10s} atoms={trace.atoms}")

    vocab = fit_bpe((t.atoms for t in traces), vocab_size=20, seed=0)
    print(
        f"\nlearned vocabulary: {len(vocab.atoms)} atoms + "
        f"{len(vocab.merges)} merges = {vocab.size} tokens"
    )

    fingerprints = encode(traces, vocab=vocab)
    print(
        f"encoded {len(fingerprints)} fingerprints; each of dimension {len(fingerprints[0].counts)}"
    )

    matrix = jsd_matrix(fingerprints, group_by="agent")
    print("\npairwise JSD by agent (upper triangle only):")
    for record in matrix.to_records():
        if str(record["row"]) < str(record["col"]):
            print(f"  {record['row']:10s} vs {record['col']:10s}  JSD = {record['jsd']:.4f}")


if __name__ == "__main__":
    main()
