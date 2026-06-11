"""Trajectory dataset ingestion: from a Hub dataset to canonical Traces.

Two ingestion paths share the canonicalization machinery:

- :func:`ingest` / :func:`plan` -- *automatic format detection*. Given only a
  dataset id, introspect its schema and sniff sample rows to infer the adapter
  and field map; no adapter argument needed. Use ``plan`` for a dry-run preview
  of the inferred plan, ``ingest`` to stream + canonicalize.
- :func:`from_hf` -- *explicit-adapter fast path*. When you already know the
  trace format, name the adapter directly and skip detection.

Importing this package also imports the built-in adapters
(:mod:`procgrep.ingest.adapters`), which self-register their format detectors
with :mod:`procgrep.canonicalize` as a side effect of import.
"""

from __future__ import annotations

from procgrep.ingest import adapters as adapters
from procgrep.ingest.core import (
    SNIFFERS,
    DatasetSchema,
    IngestionPlan,
    Sniffer,
    SniffResult,
    ingest,
    introspect,
    plan,
    sniff,
)
from procgrep.ingest.hf import from_hf

__all__ = [
    "SNIFFERS",
    "DatasetSchema",
    "IngestionPlan",
    "SniffResult",
    "Sniffer",
    "adapters",
    "from_hf",
    "ingest",
    "introspect",
    "plan",
    "sniff",
]
