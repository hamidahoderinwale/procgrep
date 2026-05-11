"""Project fingerprints into 2D with UMAP.

UMAP (McInnes et al., 2018) is the default low-dimensional projection
for visualizing the procedural space. This module is a thin wrapper
around `umap-learn` that pins seed-dependent randomness for
reproducibility and returns a structured result alongside the
hyperparameters used.

Two granularities are supported:

* Per-fingerprint projection: every trajectory becomes one 2D point.
* Per-group projection: each group's mean distribution becomes one
  2D point, with a label.

When the number of points is small (per-group projection over a
handful of cells), UMAP's defaults are inappropriate; the caller is
expected to tune ``n_neighbors`` downward. The function exposes the
hyperparameters explicitly and records them on the result so that
downstream figures can cite the exact parameters.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import normalize
from umap import UMAP

from procgrep.encode import Fingerprint


@dataclass(frozen=True)
class UmapResult:
    """A UMAP projection plus its provenance.

    Attributes:
        labels: The label for each row (trace id or group name).
        coords: ``(n, 2)`` array of 2D coordinates aligned to
            ``labels``.
        n_neighbors: UMAP hyperparameter recorded for provenance.
        min_dist: UMAP hyperparameter recorded for provenance.
        metric: Distance metric used.
        seed: Random seed used.
    """

    labels: tuple[str, ...]
    coords: npt.NDArray[np.float64]
    n_neighbors: int
    min_dist: float
    metric: str
    seed: int


def umap_project(
    fingerprints: Iterable[Fingerprint],
    *,
    granularity: str = "trace",
    n_neighbors: int = 15,
    min_dist: float = 0.25,
    metric: str = "cosine",
    seed: int = 0,
) -> UmapResult:
    """Project fingerprints to 2D coordinates.

    Args:
        fingerprints: The trajectories to project.
        granularity: ``"trace"`` for one point per trajectory; ``"group"``
            for one point per group label (using the group-mean
            distribution).
        n_neighbors: UMAP local-neighborhood size. Must be less than
            the number of points; when projecting at group
            granularity over a handful of groups, lower this.
        min_dist: UMAP minimum-distance hyperparameter.
        metric: Distance metric (``"cosine"`` is the standard choice
            for L1-normalized distributions).
        seed: Random seed for UMAP's internal RNG. Fixes the layout
            up to numerical noise.

    Returns:
        A `UmapResult` whose ``labels`` and ``coords`` rows are
        aligned.
    """
    labels, matrix = _stack(fingerprints, granularity)
    if n_neighbors >= matrix.shape[0]:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be smaller than the number of points "
            f"({matrix.shape[0]}); lower n_neighbors for small point sets"
        )

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_components=2,
    )
    coords = reducer.fit_transform(matrix)

    return UmapResult(
        labels=labels,
        coords=np.asarray(coords, dtype=np.float64),
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        seed=seed,
    )


def _stack(
    fingerprints: Iterable[Fingerprint],
    granularity: str,
) -> tuple[tuple[str, ...], npt.NDArray[np.float64]]:
    """Build the matrix UMAP consumes, plus matching row labels."""
    fps = list(fingerprints)
    if granularity == "trace":
        labels = tuple(fp.trace_id for fp in fps)
        matrix = np.stack([fp.distribution() for fp in fps], axis=0)
        return labels, normalize(matrix, norm="l1", axis=1)
    if granularity == "group":
        by_group: dict[str, list[npt.NDArray[np.float64]]] = {}
        for fp in fps:
            by_group.setdefault(fp.group, []).append(fp.distribution())
        labels = tuple(sorted(by_group))
        matrix = np.stack(
            [np.mean(np.stack(by_group[g], axis=0), axis=0) for g in labels],
            axis=0,
        )
        return labels, normalize(matrix, norm="l1", axis=1)
    raise ValueError(f"granularity must be 'trace' or 'group', got {granularity!r}")


__all__ = ["UmapResult", "umap_project"]
