"""Intent: one-shot corpus overview composing the existing pipeline
(canonicalize → fit_bpe → encode → stats/jsd) into a single readable report.
Read this when changing what `procgrep report` shows.

Design decisions:
1. Compose, never reimplement: every number comes from an existing public
   function (`fit_bpe`, `encode`, `jsd_matrix`, `stats.*`). Benefit: one
   definition per measurement across CLI, Space, and studies. Price: the
   report only moves as fast as the library surface.
2. Group by agent. Benefit: the per-model breakdown is the question a corpus
   owner asks first. Price: corpora without an agent field collapse to one
   row (the overview stays useful).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from procgrep.bpe import fit_bpe
from procgrep.encode import Fingerprint, encode
from procgrep.jsd import jsd_matrix
from procgrep.stats import (
    effective_vocab_size_per_group,
    entropies_per_group,
)
from procgrep.types import PROCEDURE_SEPARATOR, Trace


@dataclass(frozen=True)
class AgentRow:
    """Per-agent slice of the corpus overview.

    Attributes:
        n_traces: Trajectories attributed to this agent.
        median_len: Median atom-sequence length.
        entropy_median: Median per-trajectory procedure entropy (nats).
        effective_vocab: exp(entropy) of the agent-mean procedure
            distribution; diversity as an equivalent uniform vocab size.
    """

    agent: str
    n_traces: int
    median_len: float
    entropy_median: float
    effective_vocab: float


@dataclass(frozen=True)
class CorpusReport:
    """What is observed in one trajectory corpus.

    Attributes:
        atom_mix: Corpus-wide share of each atom, descending.
        top_procedures: Multi-step procedures by share of all procedure
            tokens, descending, ``(procedure, share)``.
        agents: Per-agent rows, largest first.
        jsd_pairs: Pairwise JSD between agent-mean fingerprints,
            ``(agent_a, agent_b, jsd)``, most divergent first. Empty when
            the corpus has a single agent.
        exact_duplicate_rate: ``1 - unique atom sequences / n_traces``.
        n_empty: Traces that canonicalized to zero atoms; a high count
            usually means the adapter did not match the source format.
    """

    source: str
    n_traces: int
    median_len: float
    mean_len: float
    atom_mix: list[tuple[str, float]]
    procedure_vocab_size: int
    top_procedures: list[tuple[str, float]]
    exact_duplicate_rate: float
    n_empty: int = 0
    agents: list[AgentRow] = field(default_factory=list)
    jsd_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping of every field."""
        return asdict(self)

    def summary(self) -> str:
        """Human-readable, numbers-on-show report."""
        lines = [
            f"corpus            {self.n_traces:,} traces from {self.source}",
        ]
        if self.n_empty:
            note = " (adapter mismatch? try --dry-run)" if self.n_empty == self.n_traces else ""
            lines.append(
                f"parse yield       {self.n_traces - self.n_empty:,}/{self.n_traces:,} "
                f"non-empty{note}"
            )
        lines += [
            f"length            median {self.median_len:.0f} atoms, mean {self.mean_len:.0f}",
            f"exact duplicates  {self.exact_duplicate_rate:6.1%}",
            "",
            "action mix        "
            + "  ".join(f"{atom} {share:.0%}" for atom, share in self.atom_mix[:6]),
            "",
            f"procedures        {self.procedure_vocab_size} learned; top by share:",
        ]
        lines += [f"  {share:5.1%}  {proc}" for proc, share in self.top_procedures]
        if len(self.agents) > 1:
            lines += ["", "per agent         n      med len   entropy   eff vocab"]
            lines += [
                f"  {row.agent:<24s} {row.n_traces:>5,}  {row.median_len:>7.0f}"
                f"   {row.entropy_median:>7.2f}   {row.effective_vocab:>9.1f}"
                for row in self.agents
            ]
        if self.jsd_pairs:
            lines += ["", "divergence (JSD between agent means)"]
            lines += [f"  {a} vs {b}: {v:.3f}" for a, b, v in self.jsd_pairs[:5]]
        return "\n".join(lines)


def build_report(
    traces: Sequence[Trace],
    *,
    source: str = "traces",
    vocab_size: int = 64,
    top: int = 8,
    seed: int = 0,
) -> CorpusReport:
    """Build the corpus overview for already-canonicalized traces.

    Args:
        source: Label echoed in the report (dataset id or file name).
        vocab_size: BPE procedure vocabulary size.
        top: Multi-step procedures to list.
        seed: Forwarded to `fit_bpe` for reproducible merges.

    Raises:
        ValueError: ``traces`` is empty.
    """
    if not traces:
        raise ValueError("cannot report on an empty corpus")

    # measure only what parsed; the empty count is reported, not averaged in
    n_empty = sum(1 for t in traces if not t.atoms)
    parsed = [t for t in traces if t.atoms] or list(traces)

    lengths = [len(t.atoms) for t in parsed]
    atom_counts: dict[str, int] = {}
    for t in parsed:
        for a in t.atoms:
            atom_counts[a] = atom_counts.get(a, 0) + 1
    total_atoms = sum(atom_counts.values()) or 1
    atom_mix = sorted(
        ((a, c / total_atoms) for a, c in atom_counts.items()),
        key=lambda kv: -kv[1],
    )

    vocab = fit_bpe([t.atoms for t in parsed], vocab_size=vocab_size, seed=seed)
    fingerprints = encode(parsed, vocab=vocab)
    tokens = vocab.tokens()

    token_totals = [0] * len(tokens)
    for fp in fingerprints:
        for i, c in enumerate(fp.counts):
            token_totals[i] += c
    grand_total = sum(token_totals) or 1
    top_procedures = sorted(
        (
            (tok, token_totals[i] / grand_total)
            for i, tok in enumerate(tokens)
            if PROCEDURE_SEPARATOR in tok and token_totals[i] > 0
        ),
        key=lambda kv: -kv[1],
    )[:top]

    agents = _agent_rows(parsed, fingerprints)
    jsd_pairs = _jsd_pairs(fingerprints) if len(agents) > 1 else []

    return CorpusReport(
        source=source,
        n_traces=len(traces),
        median_len=float(statistics.median(lengths)),
        mean_len=float(statistics.fmean(lengths)),
        atom_mix=atom_mix,
        procedure_vocab_size=len(tokens),
        top_procedures=top_procedures,
        exact_duplicate_rate=1.0 - len({tuple(t.atoms) for t in parsed}) / len(parsed),
        n_empty=n_empty,
        agents=agents,
        jsd_pairs=jsd_pairs,
    )


def _agent_rows(traces: Sequence[Trace], fingerprints: Sequence[Fingerprint]) -> list[AgentRow]:
    entropy = entropies_per_group(fingerprints, group_by="agent")
    eff_vocab = effective_vocab_size_per_group(fingerprints, group_by="agent")
    lengths_by_agent: dict[str, list[int]] = {}
    for t in traces:
        lengths_by_agent.setdefault(t.agent, []).append(len(t.atoms))
    rows = [
        AgentRow(
            agent=agent,
            n_traces=len(lens),
            median_len=float(statistics.median(lens)),
            entropy_median=entropy[agent].median if agent in entropy else 0.0,
            effective_vocab=eff_vocab.get(agent, 1.0),
        )
        for agent, lens in lengths_by_agent.items()
    ]
    return sorted(rows, key=lambda r: -r.n_traces)


def _jsd_pairs(fingerprints: Sequence[Fingerprint]) -> list[tuple[str, str, float]]:
    matrix = jsd_matrix(fingerprints, group_by="agent")
    seen: set[frozenset[str]] = set()
    pairs: list[tuple[str, str, float]] = []
    for rec in matrix.to_records():
        a, b, v = str(rec["row"]), str(rec["col"]), float(rec["jsd"])
        if a == b or frozenset((a, b)) in seen:
            continue
        seen.add(frozenset((a, b)))
        pairs.append((a, b, v))
    return sorted(pairs, key=lambda p: -p[2])
