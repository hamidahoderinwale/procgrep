"""Curate a trajectory corpus: structural redundancy and a diverse subset.

Large public trajectory datasets (SWE-smith 76k, nebius 80k, nvidia SWE-Hero
34k, ...) are heavily redundant in *how* the agent worked: many trajectories
repeat the same action sequence. Training-data builders dedup this by hand --
typically exact action-sequence match plus a "keep the shortest" heuristic.
This module does it structurally and reproducibly on procgrep fingerprints:

- **exact redundancy**: trajectories sharing an identical atom sequence.
- **near redundancy**: trajectories whose procedure fingerprint falls within
  ``near_dup_jsd`` Jensen-Shannon distance of an already-seen representative
  (greedy single-pass clustering, so it scales to large corpora).
- **diverse subset**: a farthest-point (max-min JSD) selection that maximizes
  procedural coverage at a target size, reported against the shortest-K and
  random-K baselines so the coverage it buys is explicit.

Nothing here runs agents or calls a model. It reads canonical ``Trace`` objects
(see :func:`procgrep.canonicalize` / :func:`procgrep.hf.from_hf`) and emits a
:class:`CurationReport`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.types import Trace

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CurationReport:
    """Structural redundancy of a corpus and a procedurally-diverse subset.

    Attributes:
        n_traces: Corpus size.
        n_exact_unique: Distinct atom sequences.
        exact_duplicate_rate: ``1 - n_exact_unique / n_traces``.
        n_near_clusters: Fingerprint clusters at ``near_dup_jsd``.
        near_duplicate_rate: ``1 - n_near_clusters / n_traces``.
        near_dup_jsd: JSD threshold used for near-duplicate clustering.
        procedure_vocab_size: Size of the induced BPE procedure vocabulary.
        subset_size: ``k`` selected for the diverse subset.
        subset_trace_ids: Trace ids of the farthest-point subset.
        coverage_diverse / coverage_shortest / coverage_random: Fraction of
            the procedure vocabulary present in each size-``k`` subset.
        mean_jsd_diverse / mean_jsd_shortest: Mean pairwise JSD within each
            subset (higher = more procedurally varied).
    """

    n_traces: int
    n_exact_unique: int
    exact_duplicate_rate: float
    n_near_clusters: int
    near_duplicate_rate: float
    near_dup_jsd: float
    procedure_vocab_size: int
    subset_size: int
    subset_indices: list[int]
    subset_trace_ids: list[str]
    coverage_diverse: float
    coverage_shortest: float
    coverage_random: float
    mean_jsd_diverse: float
    mean_jsd_shortest: float

    def summary(self) -> str:
        """Human-readable, numbers-on-show report."""
        lines = [
            f"corpus            {self.n_traces:,} traces",
            f"exact duplicates  {self.exact_duplicate_rate:6.1%}  "
            f"({self.n_exact_unique:,} unique action sequences)",
            f"near duplicates   {self.near_duplicate_rate:6.1%}  "
            f"at JSD<{self.near_dup_jsd:g}  ({self.n_near_clusters:,} procedure clusters)",
            f"procedure vocab   {self.procedure_vocab_size} tokens",
            "",
            f"diverse subset    {self.subset_size:,} traces (farthest-point, max-min JSD)",
            f"  procedure coverage   diverse {self.coverage_diverse:6.1%}   "
            f"shortest {self.coverage_shortest:6.1%}   random {self.coverage_random:6.1%}",
            f"  mean pairwise JSD    diverse {self.mean_jsd_diverse:.3f}    "
            f"shortest {self.mean_jsd_shortest:.3f}",
            f"→ shortest-K keeps {self.coverage_shortest:.0%} of procedures; "
            f"diversity sampling keeps {self.coverage_diverse:.0%}.",
        ]
        return "\n".join(lines)


def _jsd_to_rows(rows: FloatArray, q: FloatArray) -> FloatArray:
    """Base-2 Jensen-Shannon distance of ``q`` against every row of ``rows``.

    ``rows`` and ``q`` are L1-normalized distributions over the same support.
    """
    m = 0.5 * (rows + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_rows = np.where(rows > 0, rows * np.log2(rows / m), 0.0).sum(axis=1)
        # q broadcasts against m (n, V); sum over axis=1 for a per-row KL.
        kl_q = np.where(q > 0, q * np.log2(q / m), 0.0).sum(axis=1)
    result: FloatArray = 0.5 * kl_rows + 0.5 * kl_q
    return result


def _distributions(
    traces: Sequence[Trace], *, vocab_size: int, seed: int
) -> tuple[int, FloatArray, FloatArray]:
    """Induce a BPE vocabulary and return (vocab_size, distributions, counts)."""
    vocab = fit_bpe((t.atoms for t in traces), vocab_size=vocab_size, seed=seed)
    fingerprints = list(encode(traces, vocab=vocab))
    counts = np.array([list(fp.counts) for fp in fingerprints], dtype=float)
    # Force L1 normalization so every row is a proper distribution (JSD <= 1),
    # regardless of how Fingerprint.distribution() handles empty trajectories.
    row_sums = counts.sum(axis=1, keepdims=True)
    dists = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return vocab.size, dists, counts


def _greedy_near_clusters(dists: FloatArray, eps: float) -> list[int]:
    """Single-pass greedy clustering; returns representative row indices.

    Each trace joins the first existing cluster within ``eps`` JSD, else opens
    a new one. O(n * n_clusters) -- cheap when the corpus is redundant.
    """
    if len(dists) == 0:
        return []
    reps: list[int] = [0]
    rep_mat = dists[0:1]
    for i in range(1, len(dists)):
        if float(_jsd_to_rows(rep_mat, dists[i]).min()) >= eps:
            reps.append(i)
            rep_mat = np.vstack([rep_mat, dists[i]])
    return reps


def _farthest_point(dists: FloatArray, k: int, *, seed: int) -> list[int]:
    """Greedy max-min (farthest-point) selection of ``k`` row indices."""
    n = len(dists)
    k = min(k, n)
    if k == 0:
        return []
    start = int(np.random.RandomState(seed).randint(n))
    selected = [start]
    min_dist = _jsd_to_rows(dists, dists[start])
    while len(selected) < k:
        min_dist[selected] = -1.0  # never re-select an already-chosen point
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] <= 0.0:
            break  # every remaining point is a duplicate of one already chosen
        selected.append(nxt)
        min_dist = np.minimum(min_dist, _jsd_to_rows(dists, dists[nxt]))
    return selected


def _coverage(counts_subset: FloatArray) -> float:
    """Fraction of procedure-vocabulary tokens present in the subset."""
    if counts_subset.size == 0:
        return 0.0
    present = int((counts_subset.sum(axis=0) > 0).sum())
    return float(present / counts_subset.shape[1])


def _mean_pairwise_jsd(dists_subset: FloatArray) -> float:
    """Mean off-diagonal pairwise JSD within a (small) subset."""
    n = len(dists_subset)
    if n < 2:
        return 0.0
    total = sum(float(_jsd_to_rows(dists_subset, dists_subset[i]).sum()) for i in range(n))
    return total / (n * n - n)


def curate(
    traces: Iterable[Trace],
    *,
    vocab_size: int = 128,
    seed: int = 0,
    near_dup_jsd: float = 0.05,
    target_size: int | None = None,
) -> CurationReport:
    """Measure redundancy and select a procedurally-diverse subset.

    Args:
        traces: Canonical traces.
        vocab_size: BPE procedure vocabulary size.
        seed: Provenance seed for BPE and the farthest-point start.
        near_dup_jsd: JSD threshold for near-duplicate clustering.
        target_size: Diverse-subset size; defaults to the near-dup cluster
            count (the "deduplicated" size).
    """
    trace_list = list(traces)
    n = len(trace_list)
    if n == 0:
        raise ValueError("curate requires a non-empty corpus")
    if not any(t.atoms for t in trace_list):
        raise ValueError(
            "every trajectory canonicalized to empty atoms — the adapter likely "
            "does not match this dataset's format (check `procgrep curate --dry-run`)"
        )

    n_exact_unique = len({tuple(t.atoms) for t in trace_list})
    vocab_size_actual, dists, counts = _distributions(trace_list, vocab_size=vocab_size, seed=seed)
    reps = _greedy_near_clusters(dists, near_dup_jsd)
    n_near = len(reps)
    k = target_size if target_size is not None else n_near

    diverse_idx = _farthest_point(dists, k, seed=seed)
    lengths = np.array([len(t.atoms) for t in trace_list])
    shortest_idx = list(np.argsort(lengths, kind="stable")[:k])
    rng = np.random.RandomState(seed)
    random_idx = list(rng.choice(n, size=min(k, n), replace=False))

    return CurationReport(
        n_traces=n,
        n_exact_unique=n_exact_unique,
        exact_duplicate_rate=1.0 - n_exact_unique / n,
        n_near_clusters=n_near,
        near_duplicate_rate=1.0 - n_near / n,
        near_dup_jsd=near_dup_jsd,
        procedure_vocab_size=vocab_size_actual,
        subset_size=len(diverse_idx),
        subset_indices=[int(i) for i in diverse_idx],
        subset_trace_ids=[trace_list[i].trace_id for i in diverse_idx],
        coverage_diverse=_coverage(counts[diverse_idx]),
        coverage_shortest=_coverage(counts[shortest_idx]),
        coverage_random=_coverage(counts[random_idx]),
        mean_jsd_diverse=_mean_pairwise_jsd(dists[diverse_idx]),
        mean_jsd_shortest=_mean_pairwise_jsd(dists[shortest_idx]),
    )


__all__ = ["CurationReport", "curate"]
