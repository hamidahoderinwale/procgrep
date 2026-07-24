"""Learn a BPE procedure vocabulary over canonical atom sequences.

Byte-Pair Encoding (Sennrich et al., 2016) applied to atom-level
sequences: the base alphabet is the corpus's unique atoms; each merge
glues the most frequent adjacent pair into a new procedure token.
Output is a `ProcedureVocabulary` (atoms + ordered merges), serializable
to JSON and consumed by `encode`, `jsd`, `umap_project`, `probe`, and
`patterns`. Ties on pair frequency break lexicographically for
deterministic output.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from procgrep.types import PROCEDURE_SEPARATOR, Atom, AtomSequence


@dataclass(frozen=True)
class ProcedureVocabulary:
    """A learned BPE procedure vocabulary.

    Attributes:
        atoms: Base alphabet, sorted lexicographically.
        merges: Ordered ``(left, right)`` pairs. Applying a merge
            replaces adjacent ``left, right`` with
            ``left + PROCEDURE_SEPARATOR + right``.
        seed: Recorded for provenance. The algorithm is deterministic,
            so the seed has no effect on the result given fixed inputs.
        min_pair_frequency: Frequency floor used during training.
    """

    atoms: tuple[Atom, ...]
    merges: tuple[tuple[str, str], ...]
    seed: int
    min_pair_frequency: int

    @property
    def size(self) -> int:
        """Atoms plus merged procedures."""
        return len(self.atoms) + len(self.merges)

    def tokens(self) -> tuple[str, ...]:
        """Tokens in canonical order: atoms first, then merges.

        A token's position here is its index in encoded fingerprints
        and is stable across loads.
        """
        merged = tuple(_join(left, right) for left, right in self.merges)
        return self.atoms + merged

    def index(self) -> dict[str, int]:
        """Token to index map for encoding."""
        return {t: i for i, t in enumerate(self.tokens())}


def fit_bpe(
    sequences: Iterable[AtomSequence],
    *,
    vocab_size: int,
    seed: int = 0,
    min_pair_frequency: int = 2,
) -> ProcedureVocabulary:
    """Learn a BPE procedure vocabulary from atom sequences.

    Args:
        vocab_size: Target size (atoms + merges). Must be at least the
            number of unique atoms in the corpus.
        min_pair_frequency: Stop merging once the top pair drops below
            this floor.

    Returns:
        A `ProcedureVocabulary` with ``size <= vocab_size``. May stop early
        when no pair clears the frequency floor.

    Raises:
        ValueError: If ``vocab_size`` is below the corpus's unique-atom
            count.
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

    return ProcedureVocabulary(
        atoms=atoms,
        merges=tuple(merges),
        seed=seed,
        min_pair_frequency=min_pair_frequency,
    )


def apply_vocab(seq: AtomSequence, vocab: ProcedureVocabulary) -> list[str]:
    """Tokenize an atom sequence using a learned vocabulary's merges."""
    out = list(seq)
    for pair in vocab.merges:
        out = _apply_merge(out, pair)
    return out


def save_vocab(vocab: ProcedureVocabulary, path: Path) -> None:
    """Write a vocabulary to JSON at ``path``."""
    payload = {
        "atoms": list(vocab.atoms),
        "merges": [list(m) for m in vocab.merges],
        "seed": vocab.seed,
        "min_pair_frequency": vocab.min_pair_frequency,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def load_vocab(path: Path) -> ProcedureVocabulary:
    """Load a vocabulary written by `save_vocab`."""
    payload = json.loads(path.read_text())
    return ProcedureVocabulary(
        atoms=tuple(payload["atoms"]),
        merges=tuple((m[0], m[1]) for m in payload["merges"]),
        seed=int(payload["seed"]),
        min_pair_frequency=int(payload["min_pair_frequency"]),
    )


def _join(left: str, right: str) -> str:
    """Glue two tokens into one procedure token."""
    return left + PROCEDURE_SEPARATOR + right


def _most_frequent_pair(sequences: list[list[str]], min_frequency: int) -> tuple[str, str] | None:
    """Return the top adjacent pair, or None if none clears ``min_frequency``.

    Ties break lexicographically.
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
    """Greedy left-to-right merge of ``pair`` in one sequence."""
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


def render_vocab_tree(vocab: ProcedureVocabulary) -> str:
    """Render the vocabulary as merge trees: the procedure hierarchy.

    BPE builds procedures bottom-up, gluing the most frequent adjacent pair
    into a new token, so every merged procedure decomposes into the two tokens
    it came from -- recursively, down to atoms. This renders that derivation:
    atoms are the leaves, and the *maximal* procedures (those never merged into
    a larger token) are the roots, so you can see which sub-procedures recur and
    what they compose into. Indentation is merge depth.
    """
    atoms = set(vocab.atoms)
    children: dict[str, tuple[str, str]] = {}
    used: set[str] = set()
    for left, right in vocab.merges:
        children[_join(left, right)] = (left, right)
        used.add(left)
        used.add(right)
    roots = [
        token for token in (_join(left, right) for left, right in vocab.merges) if token not in used
    ]

    lines: list[str] = [
        f"{len(vocab.atoms)} atoms: {', '.join(sorted(vocab.atoms))}",
        f"{len(vocab.merges)} merges, {len(roots)} maximal procedures:",
    ]

    def walk(token: str, depth: int) -> None:
        indent = "  " * depth
        if token in atoms:
            lines.append(f"{indent}{token}")
            return
        lines.append(f"{indent}{token.replace(PROCEDURE_SEPARATOR, ' -> ')}")
        left, right = children[token]
        walk(left, depth + 1)
        walk(right, depth + 1)

    for root in roots:
        lines.append("")
        walk(root, 0)
    return "\n".join(lines)


__all__ = [
    "ProcedureVocabulary",
    "apply_vocab",
    "fit_bpe",
    "load_vocab",
    "render_vocab_tree",
    "save_vocab",
]
