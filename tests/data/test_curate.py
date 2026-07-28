"""Tests for `procgrep.curate` (structural redundancy + diverse subset)."""

from __future__ import annotations

import pytest

from procgrep.curate import CurationReport, curate
from procgrep.types import Trace


def _corpus() -> list[Trace]:
    seqs = {
        "t1": ["read_file", "edit", "run_test", "submit"],
        "t2": ["read_file", "edit", "run_test", "submit"],  # exact dup of t1
        "t3": ["search_repo", "read_file", "think", "edit"],
        "t4": ["think", "think", "edit", "edit", "run_test"],
        "t5": [
            "search_repo",
            "search_repo",
            "read_file",
            "read_file",
            "edit",
            "run_test",
            "submit",
        ],
        "t6": ["edit", "edit", "edit", "edit"],
    }
    return [Trace(trace_id=tid, agent="a", atoms=atoms) for tid, atoms in seqs.items()]


def test_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        curate([])


def test_all_empty_atoms_raises() -> None:
    with pytest.raises(ValueError, match="empty atoms"):
        curate([Trace(trace_id="x", agent="a", atoms=[])])


def test_exact_duplicate_rate_is_deterministic() -> None:
    report = curate(_corpus(), target_size=3, seed=0)
    assert isinstance(report, CurationReport)
    assert report.n_traces == 6
    assert report.n_exact_unique == 5  # t1 == t2
    assert report.exact_duplicate_rate == pytest.approx(1 - 5 / 6)


def test_report_ranges_and_subset_shape() -> None:
    report = curate(_corpus(), target_size=3, seed=0)
    assert report.subset_size == 3
    assert len(report.subset_trace_ids) == 3
    assert len(set(report.subset_indices)) == 3  # distinct picks
    for rate in (report.exact_duplicate_rate, report.near_duplicate_rate):
        assert 0.0 <= rate <= 1.0
    for cov in (report.coverage_diverse, report.coverage_shortest, report.coverage_random):
        assert 0.0 <= cov <= 1.0
    assert report.procedure_vocab_size > 0


def test_near_clustering_is_at_least_as_aggressive_as_exact() -> None:
    # near-duplicate clustering merges within a JSD radius, so it can only
    # collapse the corpus further than exact-match dedup.
    report = curate(_corpus(), near_dup_jsd=0.1, target_size=3, seed=0)
    assert report.n_near_clusters <= report.n_exact_unique
    assert report.near_duplicate_rate >= report.exact_duplicate_rate


def test_target_size_defaults_to_near_cluster_count() -> None:
    report = curate(_corpus(), near_dup_jsd=0.1, seed=0)
    assert report.subset_size == report.n_near_clusters


def test_deterministic_under_fixed_seed() -> None:
    a = curate(_corpus(), target_size=3, seed=0)
    b = curate(_corpus(), target_size=3, seed=0)
    assert a.subset_trace_ids == b.subset_trace_ids
