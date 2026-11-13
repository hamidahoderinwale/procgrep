"""Tests for `procgrep.umap_project`.

UMAP is non-trivial to test against golden values because of its
dependence on numpy version and BLAS implementation. The tests here
assert the structural properties (shape, label alignment, error on
too-large `n_neighbors`) rather than coordinate values.
"""

from __future__ import annotations

import numpy as np
import pytest

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.umap_project import umap_project

umap = pytest.importorskip("umap")  # umap-learn is the optional 'viz' extra; skip if absent


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


@pytest.mark.parametrize("modern", [False, True])
def test_umap_keeps_legacy_layout_on_old_and_new_versions(
    structured_corpus: list, monkeypatch: pytest.MonkeyPatch, modern: bool
) -> None:
    """New UMAP defaults must not change embeddings or break older installations."""
    used = {}

    class Reducer:
        def __init__(self, **kwargs):
            used.update(kwargs)
            self.params = {"compatibility_layout": None} if modern else {}

        def get_params(self):
            return self.params.copy()

        def set_params(self, **kwargs):
            assert modern, "0.5.x does not support compatibility_layout"
            self.params.update(kwargs)
            return self

        def fit_transform(self, matrix):
            if modern:
                assert self.params["compatibility_layout"] is True
            return np.zeros((len(matrix), 2))

    monkeypatch.setattr(umap, "UMAP", Reducer)
    vocab = fit_bpe([t.atoms for t in structured_corpus], vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    result = umap_project(fps, n_neighbors=5, seed=7)
    assert result.coords.shape == (len(fps), 2)
    assert used["random_state"] == result.seed == 7
