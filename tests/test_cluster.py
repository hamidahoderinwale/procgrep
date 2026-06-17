"""Tests for `procgrep.cluster` -- task clustering with a pluggable embedder."""

from __future__ import annotations

import numpy as np
import pytest

from procgrep.cluster import cluster_tasks, hf_embedder


def _dummy_embedder(texts: list[str]) -> np.ndarray:
    # Embed each text by its first character, so same-prefix texts coincide.
    return np.array([[float(ord(t[0])), 0.0] for t in texts])


def test_cluster_tasks_labels_each_and_groups_by_embedding() -> None:
    texts = ["apple", "ant", "banana", "berry", "cat", "car"]
    labels = cluster_tasks(texts, _dummy_embedder, k=3, seed=0)
    assert len(labels) == len(texts)
    assert len(set(labels)) == 3
    # same first letter -> same cluster
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[4] == labels[5]
    # different first letters -> different clusters
    assert labels[0] != labels[2] != labels[4]


def test_cluster_tasks_validates_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        cluster_tasks([], _dummy_embedder, k=2)
    with pytest.raises(ValueError, match="must be in"):
        cluster_tasks(["a", "b"], _dummy_embedder, k=5)


def test_cluster_tasks_checks_embedder_returns_row_per_text() -> None:
    def bad_embedder(texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts) + 1, 2))  # wrong row count

    with pytest.raises(ValueError, match="one row per text"):
        cluster_tasks(["a", "b"], bad_embedder, k=1)


def test_hf_embedder_needs_the_extra_or_is_callable() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        # The common case in this env: the extra is not installed.
        with pytest.raises(ImportError, match="sentence-transformers"):
            hf_embedder()
    else:  # pragma: no cover - only when procgrep[embed] is installed
        assert callable(hf_embedder())
