"""Tests for `procgrep.bpe`."""

from __future__ import annotations

from pathlib import Path

import pytest

from procgrep.bpe import (
    ProcedureVocabulary,
    apply_vocab,
    fit_bpe,
    load_vocab,
    save_vocab,
)
from procgrep.types import ATOM_EDIT, ATOM_RUN_TEST, PROCEDURE_SEPARATOR


def test_fit_bpe_learns_most_frequent_pair_first(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    # (edit, run_test) appears more than any other adjacent pair.
    assert vocab.merges[0] == (ATOM_EDIT, ATOM_RUN_TEST)


def test_fit_bpe_is_deterministic(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    a = fit_bpe(sequences, vocab_size=10)
    b = fit_bpe(sequences, vocab_size=10)
    assert a.atoms == b.atoms
    assert a.merges == b.merges


def test_fit_bpe_respects_min_pair_frequency(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=100, min_pair_frequency=100)
    assert vocab.merges == ()


def test_fit_bpe_rejects_vocab_size_below_alphabet(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    with pytest.raises(ValueError, match="smaller than"):
        fit_bpe(sequences, vocab_size=1)


def test_apply_vocab_glues_learned_pair(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    out = apply_vocab([ATOM_EDIT, ATOM_RUN_TEST, ATOM_EDIT, ATOM_RUN_TEST], vocab)
    merged = ATOM_EDIT + PROCEDURE_SEPARATOR + ATOM_RUN_TEST
    assert out == [merged, merged]


def test_vocab_tokens_unique_and_sized(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    tokens = vocab.tokens()
    assert len(tokens) == vocab.size
    assert len(set(tokens)) == len(tokens)


def test_vocab_round_trips_through_disk(tmp_path: Path, small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10, seed=42, min_pair_frequency=2)
    out_path = tmp_path / "vocab.json"
    save_vocab(vocab, out_path)
    loaded = load_vocab(out_path)
    assert isinstance(loaded, ProcedureVocabulary)
    assert loaded.atoms == vocab.atoms
    assert loaded.merges == vocab.merges
    assert loaded.seed == vocab.seed
    assert loaded.min_pair_frequency == vocab.min_pair_frequency
