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
- ``reward``: declarative ``ProcedureSpec`` (phases, penalties, target) that
  scores trajectories into a [0, 1] partial reward and derives from winners.
- ``program``: the programmability loop (enforce / verify / optimize) over a
  ``ProcedureSpec``. Model-free: emits enforcement artifacts, never runs agents.
- ``scaffolds``: render a ``ProcedureSpec`` into a coding agent harness's own
  customization format (SWE-agent config fragment, OpenHands Skill markdown).
- ``patterns``: YAML rule-based pattern matching over atom sequences.
- ``stats``: group-level descriptive and discriminative summaries.
- ``adapters``: trace format adapters (SWE-agent, Agentless, DARS, GumTree).

Public API is re-exported here; see README and METRICS.md for usage.
"""

from __future__ import annotations

from procgrep.bpe import ProcedureVocabulary, fit_bpe, load_vocab, render_vocab_tree, save_vocab
from procgrep.canonicalize import canonicalize, register_adapter
from procgrep.cluster import Embedder, cluster_tasks, hf_embedder
from procgrep.encode import Fingerprint, encode
from procgrep.ingest import adapters as adapters
from procgrep.ingest.adapters.gumtree import (
    gumtree_adapter,
    gumtree_atom,
    parse_gumtree_jsondiff,
    run_jsondiff,
)
from procgrep.ingest.hf import from_hf
from procgrep.jsd import JsdMatrix, jsd, jsd_matrix
from procgrep.library import ProcedureLibrary
from procgrep.lineage_diff import AxisResult, LineageDiff, lineage_diff
from procgrep.patterns import PatternReport, load_patterns, match_patterns
from procgrep.probe import ProbeResult, leave_one_group_out
from procgrep.program import (
    DecodeArtifact,
    GuardArtifact,
    OptimizeReport,
    RewardArtifact,
    VerifyReport,
    enforce,
    optimize,
    verify,
)
from procgrep.reward import (
    Penalty,
    Phase,
    ProcedureSpec,
    RewardResult,
    load_spec,
    score,
)
from procgrep.scaffolds import to_openhands_skill, to_swe_agent_config
from procgrep.stats import (
    DiscriminativeProcedure,
    GroupAtomFrequencies,
    GroupEntropyStats,
    atom_frequencies_per_group,
    discriminative_procedures,
    effective_vocab_size_per_group,
    entropies_per_group,
)
from procgrep.summary import SummaryDiff, summary_diff
from procgrep.types import Atom, AtomSequence, Trace, TraceAdapter
from procgrep.umap_project import UmapResult, umap_project

__version__ = "0.1.3"

__all__ = [
    "Atom",
    "AtomSequence",
    "AxisResult",
    "DecodeArtifact",
    "DiscriminativeProcedure",
    "Embedder",
    "Fingerprint",
    "GroupAtomFrequencies",
    "GroupEntropyStats",
    "GuardArtifact",
    "JsdMatrix",
    "LineageDiff",
    "OptimizeReport",
    "PatternReport",
    "Penalty",
    "Phase",
    "ProbeResult",
    "ProcedureLibrary",
    "ProcedureSpec",
    "ProcedureVocabulary",
    "RewardArtifact",
    "RewardResult",
    "SummaryDiff",
    "Trace",
    "TraceAdapter",
    "UmapResult",
    "VerifyReport",
    "__version__",
    "atom_frequencies_per_group",
    "canonicalize",
    "cluster_tasks",
    "discriminative_procedures",
    "effective_vocab_size_per_group",
    "encode",
    "enforce",
    "entropies_per_group",
    "fit_bpe",
    "from_hf",
    "gumtree_adapter",
    "gumtree_atom",
    "hf_embedder",
    "jsd",
    "jsd_matrix",
    "leave_one_group_out",
    "lineage_diff",
    "load_patterns",
    "load_spec",
    "load_vocab",
    "match_patterns",
    "optimize",
    "parse_gumtree_jsondiff",
    "register_adapter",
    "render_vocab_tree",
    "run_jsondiff",
    "save_vocab",
    "score",
    "summary_diff",
    "to_openhands_skill",
    "to_swe_agent_config",
    "umap_project",
    "verify",
]
