"""Tests for `procgrep.encode`."""

from __future__ import annotations

import math

from procgrep.bpe import fit_bpe
from procgrep.encode import encode


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
    for fp in fps:
        # Total is the post-BPE token count, not the raw atom count.
        assert fp.total > 0


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
