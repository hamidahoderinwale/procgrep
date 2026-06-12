"""Tests for `procgrep.jsd`."""

from __future__ import annotations

import math

import numpy as np

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.jsd import jsd, jsd_matrix


def test_jsd_is_zero_for_identical_distributions() -> None:
    p = [0.25, 0.25, 0.25, 0.25]
    assert jsd(p, p) == 0.0


def test_jsd_is_symmetric() -> None:
    p = [0.7, 0.2, 0.1]
    q = [0.1, 0.4, 0.5]
    assert math.isclose(jsd(p, q), jsd(q, p), abs_tol=1e-12)


def test_jsd_is_one_for_disjoint_one_hots_with_log2() -> None:
    p = [1.0, 0.0]
    q = [0.0, 1.0]
    assert math.isclose(jsd(p, q, base=2.0), 1.0, abs_tol=1e-9)


def test_jsd_is_bounded_by_log2_unit() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(8))
        q = rng.dirichlet(np.ones(8))
        value = jsd(p, q, base=2.0)
        assert 0.0 <= value <= 1.0 + 1e-12


def test_jsd_matrix_is_symmetric_with_zero_diagonal(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    matrix = jsd_matrix(fps, group_by="agent").to_array()
    assert matrix.shape == (2, 2)
    assert math.isclose(matrix[0, 0], 0.0, abs_tol=1e-12)
    assert math.isclose(matrix[1, 1], 0.0, abs_tol=1e-12)
    assert math.isclose(matrix[0, 1], matrix[1, 0], abs_tol=1e-12)


def test_jsd_matrix_separates_distinct_agents(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    matrix = jsd_matrix(fps, group_by="agent").to_array()
    # The editor and searcher agents use disjoint procedure palettes.
    assert matrix[0, 1] > 0.5
