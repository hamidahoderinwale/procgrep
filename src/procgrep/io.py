"""JSONL read/write helpers shared by the CLI subcommands.

The CLI commands compose by writing intermediate artifacts (canonical
traces, fingerprints, vocabularies, matrices) to disk and reading
them back. Keeping the serialization in one module rather than
inlining `json.dumps` in each subcommand keeps the format definitions
in one place and makes the wire formats explicit.

Three pairs of helpers:

* `read_jsonl` / `write_jsonl`: line-delimited JSON for trace and
  fingerprint corpora.
* `read_json` / `write_json`: single-object JSON for vocabularies,
  matrices, probe results, and UMAP coordinates.
* `traces_to_records` / `records_to_traces` and
  `fingerprints_to_records` / `records_to_fingerprints`: wire-format
  conversion for the two domain types.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from procgrep.encode import Fingerprint
from procgrep.types import Trace


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-empty line of ``path``."""
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    """Write records to ``path``, one JSON object per line.

    Returns:
        The number of records written.
    """
    count = 0
    with Path(path).open("w") as f:
        for record in records:
            f.write(json.dumps(record))
            f.write("\n")
            count += 1
    return count


def read_json(path: Path | str) -> Any:
    """Parse a single JSON object from ``path``."""
    return json.loads(Path(path).read_text())


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> None:
    """Write a single JSON object to ``path``."""
    Path(path).write_text(json.dumps(payload, indent=indent) + "\n")


def traces_to_records(traces: Iterable[Trace]) -> Iterator[dict[str, Any]]:
    """Convert `Trace` objects to JSON-friendly dictionaries."""
    for trace in traces:
        record: dict[str, Any] = {
            "trace_id": trace.trace_id,
            "agent": trace.agent,
            "atoms": list(trace.atoms),
        }
        if trace.group is not None:
            record["group"] = trace.group
        if trace.metadata:
            record["metadata"] = dict(trace.metadata)
        yield record


def records_to_traces(records: Iterable[dict[str, Any]]) -> Iterator[Trace]:
    """Reconstruct `Trace` objects from JSON-friendly dictionaries."""
    for record in records:
        yield Trace(
            trace_id=str(record["trace_id"]),
            agent=str(record["agent"]),
            atoms=list(record["atoms"]),
            group=None if "group" not in record else str(record["group"]),
            metadata=dict(record.get("metadata", {})),
        )


def fingerprints_to_records(fingerprints: Iterable[Fingerprint]) -> Iterator[dict[str, Any]]:
    """Convert `Fingerprint` objects to JSON-friendly dictionaries."""
    for fp in fingerprints:
        yield {
            "trace_id": fp.trace_id,
            "agent": fp.agent,
            "group": fp.group,
            "counts": list(fp.counts),
        }


def records_to_fingerprints(records: Iterable[dict[str, Any]]) -> Iterator[Fingerprint]:
    """Reconstruct `Fingerprint` objects from JSON-friendly dictionaries."""
    for record in records:
        yield Fingerprint(
            trace_id=str(record["trace_id"]),
            agent=str(record["agent"]),
            group=str(record["group"]),
            counts=tuple(int(c) for c in record["counts"]),
        )


__all__ = [
    "fingerprints_to_records",
    "read_json",
    "read_jsonl",
    "records_to_fingerprints",
    "records_to_traces",
    "traces_to_records",
    "write_json",
    "write_jsonl",
]
