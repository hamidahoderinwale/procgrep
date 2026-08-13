"""Jensen-Shannon divergence between procedural fingerprints.

``JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)`` where
``m = (p + q) / 2``. Log base 2 bounds JSD to ``[0, 1]``.

Public: `jsd` (scalar) and `jsd_matrix` (pairwise between group-mean
distributions). We avoid `scipy.spatial.distance.jensenshannon` because
it returns the square-root distance, not the divergence.
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
        records: One `JsdRecord` per ``(row, col)`` pair. The matrix is
            symmetric with zero diagonal; full grid is emitted for
            easy indexing.
        base: Log base used for the divergence.
        vocab_spec: Compact key (``content_hash:vocab_size``) of the
            vocabulary the fingerprints were encoded under; see
            `procgrep.bpe.VocabSpec`. Values from two matrices are only
            comparable when their keys match. ``None`` when the input
            fingerprints carried no spec.
    """

    groups: tuple[str, ...]
    records: tuple[JsdRecord, ...]
    base: float
    vocab_spec: str | None = None

    def to_array(self) -> npt.NDArray[np.float64]:
        """Dense ``(n, n)`` matrix in group order."""
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

    ``p`` and ``q`` must be non-negative with positive sum; both are
    L1-normalized internally. Result lies in ``[0, log_base(2)]`` and
    is symmetric up to floating-point error.
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

    Each group mean is the L1-renormalized average of its members'
    L1-normalized distributions. The result covers every ordered pair,
    including diagonal and lower triangle.

    Args:
        group_by: ``"group"`` uses `Fingerprint.group`; ``"agent"``
            groups by agent name instead.

    Raises:
        ValueError: The fingerprints carry two or more different
            vocabulary specs. Their distributions live over different
            token sets, so any JSD between them would be meaningless.
    """
    fingerprint_list = list(fingerprints)
    vocab_spec = _shared_vocab_spec(fingerprint_list)
    by_group: dict[str, list[npt.NDArray[np.float64]]] = {}
    for fp in fingerprint_list:
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

    return JsdMatrix(groups=groups, records=tuple(records), base=base, vocab_spec=vocab_spec)


def _shared_vocab_spec(fingerprints: Sequence[Fingerprint]) -> str | None:
    """The one vocabulary spec the fingerprints share, or None if untagged.

    Untagged fingerprints (``vocab_spec is None``, from files written
    before the spec existed) are treated as unknown and do not trip the
    guard; mixing two known-different specs raises.
    """
    specs = sorted({fp.vocab_spec for fp in fingerprints if fp.vocab_spec is not None})
    if len(specs) > 1:
        raise ValueError(
            f"fingerprints were encoded under {len(specs)} different vocabularies "
            f"({', '.join(specs)}); JSD between them is meaningless — re-encode "
            "all traces under one vocabulary before comparing"
        )
    return specs[0] if specs else None


def _normalize(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """L1-normalized copy; uniform on the all-zero edge case.

    Validates the distribution contract up front so silent-wrong
    results cannot reach published figures: a non-finite (NaN/inf) or
    negative entry would otherwise be masked by the zero-handling in
    `_kl` and yield a plausible-but-meaningless number. We reject such
    input loudly rather than coerce it.
    """
    if not np.all(np.isfinite(arr)):
        raise ValueError("distribution must be finite; got NaN or inf")
    if np.any(arr < 0.0):
        raise ValueError("distribution must be non-negative")
    total = float(arr.sum())
    if total <= 0.0:
        return np.full_like(arr, 1.0 / max(arr.size, 1))
    return np.asarray(arr / total, dtype=np.float64)


def _kl(
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
    base: float,
) -> float:
    """``KL(p || q)`` with zero-handling.

    ``p == 0`` terms contribute zero. ``q == 0, p > 0`` terms are
    masked: the JSD mixture ``m`` is positive wherever either input
    is, so this branch is unreachable from `jsd`.
    """
    mask = (p > 0) & (q > 0)
    if not mask.any():
        return 0.0
    ratio = p[mask] / q[mask]
    return float(np.sum(p[mask] * np.log(ratio) / np.log(base)))


__all__ = ["GroupBy", "JsdMatrix", "JsdRecord", "jsd", "jsd_matrix"]
