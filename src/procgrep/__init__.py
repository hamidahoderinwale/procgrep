"""Procedural fingerprinting of LLM coding-agent rollouts.

`procgrep` canonicalizes heterogeneous agent traces into a shared atom
alphabet, learns a BPE motif vocabulary, encodes trajectories as motif
distributions, and supports cross-group comparison via Jensen-Shannon
divergence, leave-one-group-out probes, UMAP projection, and pattern
matching. A `stats` module exposes group-level descriptive and
discriminative summary statistics on top of the core pipeline.

The public API is re-exported from this module; see the README for
intent, capabilities, and use-cases. Implementation modules live as
siblings (canonicalize, bpe, encode, jsd, umap_project, probe,
patterns, stats).
"""

from __future__ import annotations

from procgrep.bpe import MotifVocabulary, fit_bpe, load_vocab, save_vocab
from procgrep.canonicalize import canonicalize, register_adapter
from procgrep.encode import Fingerprint, encode
from procgrep.jsd import JsdMatrix, jsd, jsd_matrix
from procgrep.patterns import PatternReport, load_patterns, match_patterns
from procgrep.probe import ProbeResult, leave_one_group_out
from procgrep.stats import (
    DiscriminativeMotif,
    GroupAtomFrequencies,
    GroupEntropyStats,
    atom_frequencies_per_group,
    discriminative_motifs,
    effective_vocab_size_per_group,
    entropies_per_group,
)
from procgrep.types import Atom, AtomSequence, Trace, TraceAdapter
from procgrep.umap_project import UmapResult, umap_project

__version__ = "0.1.3"

__all__ = [
    "Atom",
    "AtomSequence",
    "DiscriminativeMotif",
    "Fingerprint",
    "GroupAtomFrequencies",
    "GroupEntropyStats",
    "JsdMatrix",
    "MotifVocabulary",
    "PatternReport",
    "ProbeResult",
    "Trace",
    "TraceAdapter",
    "UmapResult",
    "__version__",
    "atom_frequencies_per_group",
    "canonicalize",
    "discriminative_motifs",
    "effective_vocab_size_per_group",
    "encode",
    "entropies_per_group",
    "fit_bpe",
    "jsd",
    "jsd_matrix",
    "leave_one_group_out",
    "load_patterns",
    "load_vocab",
    "match_patterns",
    "register_adapter",
    "save_vocab",
    "umap_project",
]
