"""Tests for `procgrep.umap_project`.

UMAP is non-trivial to test against golden values because of its
dependence on numpy version and BLAS implementation. The tests here
assert the structural properties (shape, label alignment, error on
too-large `n_neighbors`) rather than coordinate values.
"""

from __future__ import annotations

import pytest

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.umap_project import umap_project

pytest.importorskip("umap")  # umap-learn is the optional 'viz' extra; skip if absent


def test_umap_trace_granularity_shape(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)

    result = umap_project(fps, granularity="trace", n_neighbors=5, seed=0)
    assert len(result.labels) == len(fps)
    assert result.coords.shape == (len(fps), 2)
    assert set(result.labels) == {fp.trace_id for fp in fps}


def test_umap_rejects_oversized_n_neighbors(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)

    with pytest.raises(ValueError, match="must be smaller"):
        umap_project(fps, granularity="trace", n_neighbors=len(fps) + 1, seed=0)


def test_umap_rejects_unknown_granularity(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)

    with pytest.raises(ValueError, match="granularity"):
        umap_project(fps, granularity="invalid", n_neighbors=3, seed=0)
