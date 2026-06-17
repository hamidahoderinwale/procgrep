"""Cluster task descriptions, with a pluggable embedding backend.

To ask "does procedure differ by task type?" you first need task *types*. The
clean signal is the task description's semantics -- independent of the procedure
you then measure, so the analysis is not circular. This module clusters
descriptions into groups using an embedder you supply.

The embedder is any ``Callable[[Sequence[str]], np.ndarray]`` returning one row
per text. That keeps the heavy embedding dependency out of procgrep's core (it
stays an optional extra) and lets you choose the backend: a local Hugging Face
model via `hf_embedder` (the default), a lighter local option (fastembed,
model2vec) wrapped in the same callable, or -- for non-sensitive corpora only --
an API embedder.

Privacy: with a local embedder the text never leaves the machine, and only the
returned cluster labels need to be kept. Retain the labels, discard the text and
the vectors. An API embedder sends text off-machine, so use it only where the
corpus is not sensitive (e.g. public benchmark task statements, never your own
prompts).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Embedder = Callable[[Sequence[str]], np.ndarray]
"""Maps texts to a ``(n, d)`` array, one row per text. Local by default."""


def cluster_tasks(
    texts: Sequence[str],
    embedder: Embedder,
    *,
    k: int = 8,
    seed: int = 0,
) -> list[int]:
    """Cluster ``texts`` into ``k`` groups by embedding similarity.

    ``embedder`` is any callable from texts to a row-per-text array; build a
    local Hugging Face one with `hf_embedder`, or pass your own. Returns one
    integer cluster label per input text. Only the labels need to be retained --
    the text and embeddings can be discarded, which is what keeps the result
    shareable.

    Raises:
        ValueError: if ``texts`` is empty or ``k`` exceeds the number of texts.
    """
    items = list(texts)
    if not items:
        raise ValueError("cluster_tasks needs at least one text")
    if k < 1 or k > len(items):
        raise ValueError(f"k={k} must be in [1, {len(items)}] for {len(items)} texts")

    from sklearn.cluster import KMeans

    embeddings = np.asarray(embedder(items), dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(items):
        raise ValueError(
            f"embedder must return one row per text: got shape {embeddings.shape} "
            f"for {len(items)} texts"
        )
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(embeddings)
    return [int(label) for label in labels]


def hf_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    normalize: bool = True,
) -> Embedder:
    """A local Hugging Face sentence-transformers embedder.

    The model runs on-device, so text never leaves the machine. ``model_name``
    is any sentence-transformers model, so the embedding model is configurable;
    swap in a lighter local backend (fastembed, model2vec) by writing your own
    callable with the same signature.

    Requires the ``embed`` extra: ``pip install procgrep[embed]``.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "hf_embedder needs sentence-transformers; install it with "
            "`pip install procgrep[embed]`, or pass your own embedder callable."
        ) from exc

    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            model.encode(list(texts), normalize_embeddings=normalize, show_progress_bar=False)
        )

    return embed


__all__ = ["Embedder", "cluster_tasks", "hf_embedder"]
