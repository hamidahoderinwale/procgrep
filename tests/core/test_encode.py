"""Tests for `procgrep.encode`."""

from __future__ import annotations

import math

import pytest

from procgrep.bpe import apply_vocab, fit_bpe
from procgrep.encode import encode
from procgrep.types import Trace


def test_encode_produces_correct_shape(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    fps = encode(small_corpus, vocab=vocab)
    assert len(fps) == len(small_corpus)
    for fp in fps:
        assert len(fp.counts) == vocab.size


def test_encode_counts_sum_to_tokenized_length(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    fps = encode(small_corpus, vocab=vocab)
    assert vocab.merges
    for trace, fp in zip(small_corpus, fps, strict=True):
        # Total is the post-BPE token count, not the raw atom count.
        assert fp.total == len(apply_vocab(trace.atoms, vocab))


def test_encode_distribution_is_l1_normalized(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    fps = encode(small_corpus, vocab=vocab)
    for fp in fps:
        assert math.isclose(fp.distribution().sum(), 1.0, abs_tol=1e-9)


def test_encode_forwards_identity_fields(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    fps = encode(small_corpus, vocab=vocab)
    for trace, fp in zip(small_corpus, fps, strict=True):
        assert fp.trace_id == trace.trace_id
        assert fp.agent == trace.agent
        assert fp.group == trace.grouping()


def test_encode_stamps_the_vocabulary_spec(small_corpus: list) -> None:
    sequences = [t.atoms for t in small_corpus]
    vocab = fit_bpe(sequences, vocab_size=10)
    fps = encode(small_corpus, vocab=vocab)
    for fp in fps:
        assert fp.vocab_spec == vocab.spec.compact()


@pytest.mark.parametrize("atoms", [["read_file", "lint", "edit"], ["lint"]])
def test_encode_rejects_actions_absent_from_the_vocabulary(atoms: list[str]) -> None:
    vocab = fit_bpe([["read_file", "edit"]], vocab_size=2)
    trace = Trace(trace_id="held-out", agent="a", atoms=atoms)
    with pytest.raises(ValueError, match=r"held-out.*lint"):
        encode([trace], vocab=vocab)


def test_encode_ignoring_unknowns_is_an_explicit_legacy_policy() -> None:
    vocab = fit_bpe([["read_file", "edit"]], vocab_size=2)
    trace = Trace(trace_id="held-out", agent="a", atoms=["read_file", "lint", "edit"])
    (fp,) = encode([trace], vocab=vocab, unknown="ignore")
    assert fp.counts == (1, 1)
    assert fp.total == 2
    assert fp.vocab_spec == vocab.spec.compact()


def test_encode_known_actions_are_identical_under_both_policies(small_corpus: list) -> None:
    vocab = fit_bpe([t.atoms for t in small_corpus], vocab_size=10)
    assert encode(small_corpus, vocab=vocab) == encode(small_corpus, vocab=vocab, unknown="ignore")


def test_encode_empty_trace_still_has_zero_counts() -> None:
    vocab = fit_bpe([["read_file", "edit"]], vocab_size=2)
    (fp,) = encode([Trace(trace_id="empty", agent="a", atoms=[])], vocab=vocab)
    assert fp.counts == (0, 0)
    assert fp.total == 0


def test_encode_rejects_invalid_unknown_policy() -> None:
    vocab = fit_bpe([["edit"]], vocab_size=1)
    with pytest.raises(ValueError, match=r"unknown.*error.*ignore"):
        encode([], vocab=vocab, unknown="drop")  # type: ignore[arg-type]
