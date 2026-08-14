"""Encode atom sequences as procedure-frequency distributions.

A `Fingerprint` is one trajectory's counts over the vocabulary tokens
plus its identity and grouping label. Counts are the canonical form;
the L1-normalized distribution is derived on demand. Token order is
held fixed across all fingerprints so JSD and probe routines can
compare distributions positionally.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from procgrep.bpe import ProcedureVocabulary, apply_vocab
from procgrep.types import Trace


@dataclass(frozen=True)
class Fingerprint:
    """One trajectory expressed as a procedure count vector.

    Attributes:
        group: Forwarded from the originating `Trace`. Falls back to
            ``agent`` when the trace had no group label.
        counts: Length ``vocab.size``, aligned to ``vocab.tokens()``.
        vocab_spec: Compact key (``content_hash:vocab_size``) of the
            vocabulary this fingerprint was encoded under; see
            `procgrep.bpe.VocabSpec`. Counts are only comparable between
            fingerprints with equal keys. ``None`` on fingerprints from
            before the spec existed.
    """

    trace_id: str
    agent: str
    group: str
    counts: tuple[int, ...]
    vocab_spec: str | None = None

    @property
    def total(self) -> int:
        """Token count (sequence length after BPE)."""
        return sum(self.counts)

    def distribution(self) -> npt.NDArray[np.float64]:
        """L1-normalized distribution over vocab tokens.

        Empty trajectories (``total == 0``) return a uniform vector so
        divergence routines do not divide by zero. Filter on ``total``
        to distinguish them.
        """
        arr = np.asarray(self.counts, dtype=np.float64)
        total = float(arr.sum())
        if total == 0.0:
            return np.full_like(arr, 1.0 / max(arr.size, 1))
        return np.asarray(arr / total, dtype=np.float64)

    def entropy(self) -> float:
        """Shannon entropy of the procedure distribution, in nats.

        Returns 0.0 when all mass is on one procedure, ``log(vocab_size)``
        for the uniform (empty-trajectory) case. See
        `procgrep.stats.entropies_per_group` for aggregates.
        """
        dist = self.distribution()
        positive = dist[dist > 0]
        if positive.size == 0:
            return 0.0
        return float(-np.sum(positive * np.log(positive)))


def encode(traces: Iterable[Trace], *, vocab: ProcedureVocabulary) -> list[Fingerprint]:
    """Encode traces under a fixed vocabulary.

    Position ``i`` in every returned fingerprint maps to
    ``vocab.tokens()[i]``. Output preserves input order.
    """
    index = vocab.index()
    size = vocab.size
    spec_key = vocab.spec.compact()
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
                vocab_spec=spec_key,
            )
        )
    return out


__all__ = ["Fingerprint", "encode"]
