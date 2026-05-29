"""Procedural fingerprinting of LLM coding-agent trajectories.

Answers one question: do two agents, holding some factors fixed and varying
others, produce structurally distinct procedures? Given trace logs, it
canonicalizes those traces into a shared atom alphabet, learns a BPE
procedure vocabulary, and encodes each trajectory as a procedure-frequency
distribution for cross-group comparison.

Post-hoc analysis only. Does not run agents, call models, or require an LLM
SDK. Reads trace files and emits structural artifacts.

Modules:

- ``bpe`` / ``encode``: vocabulary induction and trajectory encoding.
- ``jsd``: pairwise and group-level Jensen-Shannon divergence.
- ``lineage_diff``: four-axis structural comparison between two agent groups
  (vocabulary, entropy, outcome-quadrant, conditional).
- ``probe``: leave-one-group-out predictive transfer probe.
- ``reward``: score trajectories against a YAML procedural reward spec,
  returning a [0, 1] partial reward signal.
- ``patterns``: YAML rule-based pattern matching over atom sequences.
- ``stats``: group-level descriptive and discriminative summaries.
- ``adapters``: trace format adapters (SWE-agent, Agentless, DARS, GumTree).

Public API is re-exported here; see README and METRICS.md for usage.
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
from procgrep.reward import RewardResult, load_spec, score
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
    "RewardResult",
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
    "load_spec",
    "load_patterns",
    "load_vocab",
    "match_patterns",
    "parse_gumtree_jsondiff",
    "register_adapter",
    "run_jsondiff",
    "save_vocab",
    "score",
    "umap_project",
]
