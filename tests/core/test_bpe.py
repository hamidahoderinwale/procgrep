"""Tests for `procgrep.bpe`."""

from __future__ import annotations

from pathlib import Path

import pytest

from procgrep.bpe import (
    ProcedureVocabulary,
    apply_vocab,
    fit_bpe,
    load_vocab,
    render_vocab_tree,
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


def test_render_vocab_tree_shows_decomposition() -> None:
    # 'a b c' repeats, so a/b/c merge into a procedure that decomposes to atoms
    sequences = [["a", "b", "c"] * 4, ["a", "b"] * 3]
    vocab = fit_bpe(sequences, vocab_size=8, seed=0)
    tree = render_vocab_tree(vocab)
    assert tree.startswith(f"{len(vocab.atoms)} atoms:")
    assert "maximal procedures" in tree
    assert " -> " in tree  # at least one merged procedure rendered
    for atom in vocab.atoms:
        assert atom in tree  # every atom appears as a leaf


def test_render_vocab_tree_handles_no_merges() -> None:
    # vocab_size == #atoms means no merges; tree is just the atom line + header
    vocab = fit_bpe([["a", "b"]], vocab_size=2, seed=0)
    tree = render_vocab_tree(vocab)
    assert "0 merges, 0 maximal procedures" in tree


def test_spec_hash_is_deterministic_across_refits(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    a = fit_bpe(sequences, vocab_size=10)
    b = fit_bpe(sequences, vocab_size=10)
    assert a.spec.content_hash == b.spec.content_hash
    assert a.spec.compact() == b.spec.compact()


def test_spec_hash_changes_with_vocab_size() -> None:
    sequences = [["a", "b", "c", "d"] * 4]
    a = fit_bpe(sequences, vocab_size=5, min_pair_frequency=1)
    b = fit_bpe(sequences, vocab_size=7, min_pair_frequency=1)
    assert a.merges != b.merges  # different merge lists...
    assert a.spec.content_hash != b.spec.content_hash  # ...different hashes


def test_spec_hash_changes_with_fit_corpus() -> None:
    a = fit_bpe([["a", "b", "c"] * 4], vocab_size=5, min_pair_frequency=1)
    b = fit_bpe([["a", "c", "b"] * 4], vocab_size=5, min_pair_frequency=1)
    assert a.spec.content_hash != b.spec.content_hash


def test_spec_carries_size_seed_alphabet_and_merge_count(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10, seed=7, fit_corpus="unit-corpus")
    spec = vocab.spec
    assert spec.vocab_size == vocab.size
    assert spec.n_atoms == len(vocab.atoms)
    assert spec.n_merges == len(vocab.merges)
    assert spec.seed == 7
    assert spec.atoms == vocab.atoms
    assert spec.fit_corpus == "unit-corpus"
    assert spec.compact() == f"{spec.content_hash}:{vocab.size}"


def test_fit_corpus_round_trips_through_disk(tmp_path: Path, small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10, fit_corpus="swe-bench-lite")
    out_path = tmp_path / "vocab.json"
    save_vocab(vocab, out_path)
    loaded = load_vocab(out_path)
    assert loaded.fit_corpus == "swe-bench-lite"
    assert loaded.spec == vocab.spec


def test_load_vocab_without_fit_corpus_field(tmp_path: Path, small_corpus: list) -> None:
    # Vocab files written before the spec existed have no fit_corpus key.
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    out_path = tmp_path / "vocab.json"
    save_vocab(vocab, out_path)
    loaded = load_vocab(out_path)
    assert loaded.fit_corpus is None
