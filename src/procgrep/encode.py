"""Encode atom sequences as motif-frequency distributions.

A `Fingerprint` is one trajectory's procedural shape expressed as a
non-negative integer count vector over the vocabulary tokens, plus
the originating trace's identity and grouping label. Counts are
stored as the canonical representation; the L1-normalized
distribution is derived on demand.

This module is thin on purpose: `apply_vocab` from `procgrep.bpe`
does the BPE tokenization, and `Counter` does the counting. The
work here is plumbing: holding the vocabulary's token order fixed
across all fingerprints so that downstream JSD and probe routines
can compare distributions positionally.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from procgrep.bpe import MotifVocabulary, apply_vocab
from procgrep.types import Trace


@dataclass(frozen=True)
class Fingerprint:
    """One trajectory expressed as a motif count vector.

    Attributes:
        trace_id: Forwarded from the originating `Trace`.
        agent: Forwarded from the originating `Trace`.
        group: Forwarded from the originating `Trace`. Falls back to
            ``agent`` when the trace did not carry a group label.
        counts: Tuple of length ``vocab.size`` aligned to
            ``vocab.tokens()``.
    """

    trace_id: str
    agent: str
    group: str
    counts: tuple[int, ...]

    @property
    def total(self) -> int:
        """Total token count (sequence length after BPE)."""
        return sum(self.counts)

    def distribution(self) -> npt.NDArray[np.float64]:
        """Return the L1-normalized distribution over vocab tokens.

        An empty trajectory (``total == 0``) returns a uniform vector
        so that downstream divergence routines do not divide by zero.
        Callers who want to distinguish empty trajectories should
        filter on ``total`` directly.
        """
        arr = np.asarray(self.counts, dtype=np.float64)
        total = float(arr.sum())
        if total == 0.0:
            return np.full_like(arr, 1.0 / max(arr.size, 1))
        return np.asarray(arr / total, dtype=np.float64)

    def entropy(self) -> float:
        """Shannon entropy of the motif distribution, in nats.

        Returns 0.0 for a trajectory whose mass is on a single motif;
        returns ``log(vocab_size)`` for the uniform (empty-trajectory)
        case. Useful as a per-trajectory diversity score; aggregate
        statistics live in `procgrep.stats.entropies_per_group`.
        """
        dist = self.distribution()
        positive = dist[dist > 0]
        if positive.size == 0:
            return 0.0
        return float(-np.sum(positive * np.log(positive)))


def encode(traces: Iterable[Trace], *, vocab: MotifVocabulary) -> list[Fingerprint]:
    """Encode an iterable of `Trace` objects under a fixed vocabulary.

    The vocabulary's token order is preserved across all fingerprints;
    position `i` in every returned fingerprint corresponds to the same
    token, namely ``vocab.tokens()[i]``.

    Args:
        traces: The trajectories to fingerprint.
        vocab: The motif vocabulary to apply.

    Returns:
        One `Fingerprint` per input trace, in the same order.
    """
    index = vocab.index()
    size = vocab.size
    out: list[Fingerprint] = []
    for trace in traces:
        tokenized = apply_vocab(trace.atoms, vocab)
        token_counts = Counter(tokenized)
        counts = [0] * size
        for token, count in token_counts.items():
            position = index.get(token)
            if position is not None:
                counts[position] = count
        out.append(
            Fingerprint(
                trace_id=trace.trace_id,
                agent=trace.agent,
                group=trace.grouping(),
                counts=tuple(counts),
            )
        )
    return out


__all__ = ["Fingerprint", "encode"]
