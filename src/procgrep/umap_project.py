"""Project fingerprints into 2D with UMAP.

Thin wrapper around `umap-learn` that pins the random seed and records
hyperparameters on the result. Granularity is ``"trace"`` (one point
per trajectory) or ``"group"`` (one point per group mean). Small
point sets need a lower ``n_neighbors`` than the UMAP default.
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
        labels: Row label (trace id or group name).
        coords: ``(n, 2)`` array aligned to ``labels``.
        n_neighbors, min_dist, metric, seed: UMAP hyperparameters.
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
        granularity: ``"trace"`` (one point per trajectory) or
            ``"group"`` (one point per group mean).
        n_neighbors: Must be less than the number of points. Lower
            this for small point sets.
        metric: ``"cosine"`` is the standard choice for L1-normalized
            distributions.
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
    """Build the matrix UMAP consumes plus matching row labels."""
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
