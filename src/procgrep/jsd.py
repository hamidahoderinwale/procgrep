"""Jensen-Shannon divergence between procedural fingerprints.

Jensen-Shannon divergence is the symmetric, bounded relative of
Kullback-Leibler divergence. For two probability distributions ``p``
and ``q`` and their pointwise mixture ``m = (p + q) / 2``:

    JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)

With log base 2, JSD is bounded in ``[0, 1]``; with the natural log,
in ``[0, ln 2]``. This module defaults to log base 2 so the matrix
values are directly comparable across corpora and across vocabulary
sizes.

Two public functions:

* `jsd(p, q)`: a scalar between two distributions.
* `jsd_matrix(fingerprints, group_by=...)`: pairwise JSD between
  group-mean distributions. The result is a flat list of
  ``(row, col, jsd)`` records plus an ordered list of group names,
  which serializes cleanly to JSON for downstream use.

We deliberately avoid `scipy.spatial.distance.jensenshannon` because
that function returns the Jensen-Shannon *distance* (square root of
the divergence), not the divergence itself. Mixing the two in a
research codebase is a known source of bugs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from procgrep.encode import Fingerprint

GroupBy = Literal["agent", "group"]


@dataclass(frozen=True)
class JsdRecord:
    """One cell of a JSD matrix."""

    row: str
    col: str
    jsd: float


@dataclass(frozen=True)
class JsdMatrix:
    """Pairwise JSD between group-mean fingerprints.

    Attributes:
        groups: Group names in canonical row/column order.
        records: One `JsdRecord` per (row, col) pair. The full matrix
            is symmetric with zero diagonal; we emit all entries
            (including the diagonal and the lower triangle) for ease
            of indexing.
        base: Log base used for the divergence (2 by default).
    """

    groups: tuple[str, ...]
    records: tuple[JsdRecord, ...]
    base: float

    def to_array(self) -> npt.NDArray[np.float64]:
        """Materialize a dense ``(n, n)`` matrix in group order."""
        n = len(self.groups)
        idx = {g: i for i, g in enumerate(self.groups)}
        arr = np.zeros((n, n), dtype=np.float64)
        for record in self.records:
            arr[idx[record.row], idx[record.col]] = record.jsd
        return arr

    def to_records(self) -> list[dict[str, str | float]]:
        """JSON-friendly list-of-dicts representation."""
        return [{"row": r.row, "col": r.col, "jsd": r.jsd} for r in self.records]


def jsd(
    p: Sequence[float] | npt.NDArray[np.float64],
    q: Sequence[float] | npt.NDArray[np.float64],
    *,
    base: float = 2.0,
) -> float:
    """Jensen-Shannon divergence between two distributions.

    Args:
        p: First distribution. Must be non-negative and sum to a
            positive value; will be L1-normalized internally.
        q: Second distribution; same constraints.
        base: Logarithm base. Defaults to 2 so the result lies in
            ``[0, 1]``.

    Returns:
        A float in ``[0, log_base(2)]``. The function is symmetric:
        ``jsd(p, q) == jsd(q, p)`` up to floating-point error.
    """
    p_arr = _normalize(np.asarray(p, dtype=np.float64))
    q_arr = _normalize(np.asarray(q, dtype=np.float64))
    m = 0.5 * (p_arr + q_arr)
    return float(0.5 * _kl(p_arr, m, base) + 0.5 * _kl(q_arr, m, base))


def jsd_matrix(
    fingerprints: Iterable[Fingerprint],
    *,
    group_by: GroupBy = "group",
    base: float = 2.0,
) -> JsdMatrix:
    """Pairwise JSD between group-mean fingerprints.

    Each group's mean distribution is the L1-renormalized average of
    its members' L1-normalized distributions. The result is a
    `JsdMatrix` containing every ordered pair, including diagonal
    (zero) and lower-triangle entries.

    Args:
        fingerprints: The fingerprints to group and compare.
        group_by: ``"group"`` to use `Fingerprint.group` (the standard
            choice), or ``"agent"`` to override and group by agent
            name. Most projects assign `group` upstream when
            canonicalizing, so the default works in the common case.
        base: Logarithm base passed through to `jsd`.

    Returns:
        A `JsdMatrix` in canonical (sorted) group order.
    """
    by_group: dict[str, list[npt.NDArray[np.float64]]] = {}
    for fp in fingerprints:
        key = fp.agent if group_by == "agent" else fp.group
        by_group.setdefault(key, []).append(fp.distribution())

    groups = tuple(sorted(by_group))
    means: dict[str, npt.NDArray[np.float64]] = {}
    for g in groups:
        stacked = np.stack(by_group[g], axis=0)
        means[g] = _normalize(stacked.mean(axis=0))

    records: list[JsdRecord] = []
    for row in groups:
        for col in groups:
            value = 0.0 if row == col else jsd(means[row], means[col], base=base)
            records.append(JsdRecord(row=row, col=col, jsd=value))

    return JsdMatrix(groups=groups, records=tuple(records), base=base)


def _normalize(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return an L1-normalized copy; uniform on the all-zero edge case."""
    total = float(arr.sum())
    if total <= 0.0:
        return np.full_like(arr, 1.0 / max(arr.size, 1))
    return np.asarray(arr / total, dtype=np.float64)


def _kl(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    base: float,
) -> float:
    """KL divergence ``KL(p || q)`` with zero-handling.

    Terms where ``p == 0`` contribute zero (by convention 0 log 0 = 0).
    Terms where ``q == 0`` and ``p > 0`` would diverge; we mask them
    out because in practice the mixture ``m`` in JSD is positive
    wherever either ``p`` or ``q`` is.
    """
    mask = (p > 0) & (q > 0)
    if not mask.any():
        return 0.0
    ratio = p[mask] / q[mask]
    return float(np.sum(p[mask] * np.log(ratio) / np.log(base)))


__all__ = ["GroupBy", "JsdMatrix", "JsdRecord", "jsd", "jsd_matrix"]
