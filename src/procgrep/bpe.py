"""Learn a BPE motif vocabulary over canonical atom sequences.

Byte-Pair Encoding (Sennrich et al., 2016) was originally designed
for subword tokenization. Here we apply it to atom-level sequences:
the base alphabet is the set of canonical atoms observed in the
corpus, and each merge step glues the most frequent adjacent pair
into a new motif token.

The output is a `MotifVocabulary`: an ordered list of base atoms
plus an ordered list of merge operations. The vocabulary is the
single source of truth that the rest of the pipeline (`encode`,
`jsd`, `umap_project`, `probe`, `patterns`) consumes. It serializes
to JSON for reproducibility.

The algorithm is the classical greedy one:

1. Count adjacent-pair frequencies across all sequences.
2. Pick the most frequent pair (ties broken lexicographically for
   determinism).
3. Stop if the best pair occurs fewer than `min_pair_frequency`
   times, or if the target vocabulary size is reached.
4. Replace every occurrence of the pair in every sequence with a
   merged token, record the merge, and repeat.

The only subtlety is the merge-application order: within a single
training step we left-to-right scan each sequence and merge greedily,
the same way BPE is applied at inference time. This avoids the
"overlapping pairs" ambiguity.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from procgrep.types import MOTIF_SEPARATOR, Atom, AtomSequence


@dataclass(frozen=True)
class MotifVocabulary:
    """A learned BPE motif vocabulary.

    Attributes:
        atoms: Base alphabet, sorted lexicographically.
        merges: Ordered tuple of merge operations. Each entry is a
            ``(left, right)`` pair; applying the merge replaces every
            adjacent occurrence of ``left`` followed by ``right`` with
            the joined token ``left + MOTIF_SEPARATOR + right``.
        seed: Seed recorded for provenance. The fitting algorithm is
            deterministic, so the seed is informational only; it has
            no effect on the learned vocabulary given fixed inputs.
        min_pair_frequency: Frequency floor used during training.
    """

    atoms: tuple[Atom, ...]
    merges: tuple[tuple[str, str], ...]
    seed: int
    min_pair_frequency: int

    @property
    def size(self) -> int:
        """Total vocabulary size: atoms plus merged motifs."""
        return len(self.atoms) + len(self.merges)

    def tokens(self) -> tuple[str, ...]:
        """All tokens in canonical order: atoms first, then merges.

        The position of each token in this tuple is its index in
        encoded fingerprints. Stable across calls and across loads
        from disk.
        """
        merged = tuple(_join(left, right) for left, right in self.merges)
        return self.atoms + merged

    def index(self) -> dict[str, int]:
        """Token -> index mapping for fast lookup during encoding."""
        return {t: i for i, t in enumerate(self.tokens())}


def fit_bpe(
    sequences: Iterable[AtomSequence],
    *,
    vocab_size: int,
    seed: int = 0,
    min_pair_frequency: int = 2,
) -> MotifVocabulary:
    """Learn a BPE motif vocabulary from canonical atom sequences.

    Args:
        sequences: Iterable of canonical atom sequences (one per
            trajectory). Will be materialized into a list internally.
        vocab_size: Target vocabulary size (base atoms + merges).
            Must be at least the number of unique atoms in the corpus.
        seed: Seed recorded on the returned vocabulary for provenance.
            The algorithm is deterministic for fixed inputs.
        min_pair_frequency: Stop merging once the most frequent pair
            falls below this threshold.

    Returns:
        A `MotifVocabulary` with ``size <= vocab_size`` (the algorithm
        may stop early if the corpus lacks pairs above the frequency
        floor).

    Raises:
        ValueError: If ``vocab_size`` is less than the number of unique
            atoms in the input corpus.
    """
    materialized: list[list[str]] = [list(s) for s in sequences]
    atoms = tuple(sorted({a for s in materialized for a in s}))

    if vocab_size < len(atoms):
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than the {len(atoms)} unique "
            "atoms in the corpus; raise vocab_size or shrink the alphabet"
        )

    merges: list[tuple[str, str]] = []
    n_merges_target = vocab_size - len(atoms)

    for _ in range(n_merges_target):
        pair = _most_frequent_pair(materialized, min_pair_frequency)
        if pair is None:
            break
        merges.append(pair)
        materialized = [_apply_merge(seq, pair) for seq in materialized]

    return MotifVocabulary(
        atoms=atoms,
        merges=tuple(merges),
        seed=seed,
        min_pair_frequency=min_pair_frequency,
    )


def apply_vocab(seq: AtomSequence, vocab: MotifVocabulary) -> list[str]:
    """Apply a learned vocabulary's merges to a single atom sequence.

    Returns the tokenized sequence (atoms or merged motifs), suitable
    for downstream counting by `procgrep.encode.encode`.
    """
    out = list(seq)
    for pair in vocab.merges:
        out = _apply_merge(out, pair)
    return out


def save_vocab(vocab: MotifVocabulary, path: Path) -> None:
    """Serialize a vocabulary to JSON at ``path``."""
    payload = {
        "atoms": list(vocab.atoms),
        "merges": [list(m) for m in vocab.merges],
        "seed": vocab.seed,
        "min_pair_frequency": vocab.min_pair_frequency,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def load_vocab(path: Path) -> MotifVocabulary:
    """Load a vocabulary previously written by `save_vocab`."""
    payload = json.loads(path.read_text())
    return MotifVocabulary(
        atoms=tuple(payload["atoms"]),
        merges=tuple((m[0], m[1]) for m in payload["merges"]),
        seed=int(payload["seed"]),
        min_pair_frequency=int(payload["min_pair_frequency"]),
    )


def _join(left: str, right: str) -> str:
    """Glue two tokens into one BPE motif token."""
    return left + MOTIF_SEPARATOR + right


def _most_frequent_pair(sequences: list[list[str]], min_frequency: int) -> tuple[str, str] | None:
    """Find the most frequent adjacent pair across all sequences.

    Returns None if no pair meets ``min_frequency``. Ties on count are
    broken by lexicographic order on the pair (left, right) for
    deterministic output.
    """
    counts: Counter[tuple[str, str]] = Counter()
    for seq in sequences:
        for i in range(len(seq) - 1):
            counts[(seq[i], seq[i + 1])] += 1
    if not counts:
        return None
    best_pair, best_count = min(
        counts.items(),
        key=lambda kv: (-kv[1], kv[0]),
    )
    if best_count < min_frequency:
        return None
    return best_pair


def _apply_merge(seq: list[str], pair: tuple[str, str]) -> list[str]:
    """Left-to-right greedy merge of ``pair`` in a single sequence."""
    left, right = pair
    merged = _join(left, right)
    out: list[str] = []
    i = 0
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == left and seq[i + 1] == right:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


__all__ = [
    "MotifVocabulary",
    "apply_vocab",
    "fit_bpe",
    "load_vocab",
    "save_vocab",
]
