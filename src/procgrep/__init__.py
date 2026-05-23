"""Procedural fingerprinting of LLM coding-agent trajectories.

Canonicalizes agent traces into a shared atom alphabet, learns a BPE
procedure vocabulary, encodes trajectories as procedure distributions,
and compares groups via Jensen-Shannon divergence, leave-one-group-out
probes, UMAP projection, and pattern matching. `stats` adds group-level
descriptive and discriminative summaries.

Public API is re-exported here; see the README for usage.
"""

from __future__ import annotations

from procgrep import adapters as adapters
from procgrep.adapters.gumtree import (
    gumtree_adapter,
    gumtree_atom,
    parse_gumtree_jsondiff,
    run_jsondiff,
)
from procgrep.bpe import ProcedureVocabulary, fit_bpe, load_vocab, save_vocab
from procgrep.canonicalize import canonicalize, register_adapter
from procgrep.encode import Fingerprint, encode
from procgrep.jsd import JsdMatrix, jsd, jsd_matrix
from procgrep.lineage_diff import AxisResult, LineageDiff, lineage_diff
from procgrep.patterns import PatternReport, load_patterns, match_patterns
from procgrep.probe import ProbeResult, leave_one_group_out
from procgrep.stats import (
    DiscriminativeProcedure,
    GroupAtomFrequencies,
    GroupEntropyStats,
    atom_frequencies_per_group,
    discriminative_procedures,
    effective_vocab_size_per_group,
    entropies_per_group,
)
from procgrep.types import Atom, AtomSequence, Trace, TraceAdapter
from procgrep.umap_project import UmapResult, umap_project

__version__ = "0.1.3"

__all__ = [
    "Atom",
    "AtomSequence",
    "AxisResult",
    "DiscriminativeProcedure",
    "Fingerprint",
    "GroupAtomFrequencies",
    "GroupEntropyStats",
    "JsdMatrix",
    "LineageDiff",
    "PatternReport",
    "ProbeResult",
    "ProcedureVocabulary",
    "Trace",
    "TraceAdapter",
    "UmapResult",
    "__version__",
    "atom_frequencies_per_group",
    "canonicalize",
    "discriminative_procedures",
    "effective_vocab_size_per_group",
    "encode",
    "entropies_per_group",
    "fit_bpe",
    "gumtree_adapter",
    "gumtree_atom",
    "jsd",
    "jsd_matrix",
    "leave_one_group_out",
    "lineage_diff",
    "load_patterns",
    "load_vocab",
    "match_patterns",
    "parse_gumtree_jsondiff",
    "register_adapter",
    "run_jsondiff",
    "save_vocab",
    "umap_project",
]
