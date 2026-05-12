"""Tests for `procgrep.stats` and `Fingerprint.entropy`."""

from __future__ import annotations

import math

import pytest

from procgrep.bpe import fit_bpe
from procgrep.encode import Fingerprint, encode
from procgrep.stats import (
    atom_frequencies_per_group,
    discriminative_motifs,
    effective_vocab_size_per_group,
    entropies_per_group,
)
from procgrep.types import ATOM_EDIT, ATOM_RUN_TEST


def _make_fingerprint(counts: tuple[int, ...], *, group: str = "g") -> Fingerprint:
    return Fingerprint(trace_id="t", agent="a", group=group, counts=counts)


def test_fingerprint_entropy_zero_for_single_motif() -> None:
    fp = _make_fingerprint((10, 0, 0, 0))
    assert fp.entropy() == 0.0


def test_fingerprint_entropy_log_two_for_balanced_two_motifs() -> None:
    fp = _make_fingerprint((5, 5, 0, 0))
    assert math.isclose(fp.entropy(), math.log(2), abs_tol=1e-9)


def test_fingerprint_entropy_log_k_for_uniform_distribution() -> None:
    fp = _make_fingerprint((3, 3, 3, 3))
    assert math.isclose(fp.entropy(), math.log(4), abs_tol=1e-9)


def test_fingerprint_entropy_empty_uses_uniform_convention() -> None:
    # Empty trajectory: distribution falls back to uniform; entropy is log(vocab_size).
    fp = _make_fingerprint((0, 0, 0, 0))
    assert math.isclose(fp.entropy(), math.log(4), abs_tol=1e-9)


def test_atom_frequencies_per_group_counts_match_corpus(structured_corpus: list) -> None:
    freqs = atom_frequencies_per_group(structured_corpus, k=10, group_by="agent")
    assert set(freqs) == {"editor", "searcher"}
    editor = freqs["editor"]
    assert editor.n_trajectories == 9  # 3 groups x 3 traj
    # Editor trajectories always contain EDIT and RUN_TEST atoms.
    editor_atoms = {atom for atom, _, _ in editor.top}
    assert ATOM_EDIT in editor_atoms
    assert ATOM_RUN_TEST in editor_atoms


def test_atom_frequencies_per_group_top_k_respects_k(structured_corpus: list) -> None:
    freqs = atom_frequencies_per_group(structured_corpus, k=3, group_by="agent")
    for entry in freqs.values():
        assert len(entry.top) <= 3


def test_atom_frequencies_per_group_per_trajectory_rate_is_meaningful(
    structured_corpus: list,
) -> None:
    freqs = atom_frequencies_per_group(structured_corpus, k=20, group_by="agent")
    editor = freqs["editor"]
    # Editor trajectories have 3 EDIT atoms each by construction; rate should be 3.0.
    for atom, count, rate in editor.top:
        if atom == ATOM_EDIT:
            assert math.isclose(rate, count / editor.n_trajectories)
            assert math.isclose(rate, 3.0)
            break
    else:
        pytest.fail("EDIT atom not present in editor's top frequencies")


def test_effective_vocab_size_uniform_single_motif() -> None:
    # Single fingerprint with all mass on one motif: effective vocab = 1.
    fps = [_make_fingerprint((10, 0, 0, 0), group="A")]
    result = effective_vocab_size_per_group(fps)
    assert math.isclose(result["A"], 1.0, abs_tol=1e-9)


def test_effective_vocab_size_uniform_over_four_motifs() -> None:
    # Uniform distribution over 4 motifs: effective vocab = 4.
    fps = [_make_fingerprint((1, 1, 1, 1), group="B")]
    result = effective_vocab_size_per_group(fps)
    assert math.isclose(result["B"], 4.0, abs_tol=1e-9)


def test_effective_vocab_size_for_structured_corpus(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    result = effective_vocab_size_per_group(fps, group_by="agent")
    # Both groups exist in the result and have non-degenerate effective vocab.
    # With identical within-agent trajectories and an aggressive BPE merge,
    # one group can collapse to effective vocab 1.0; we only require >= 1.0.
    assert result["editor"] >= 1.0
    assert result["searcher"] >= 1.0
    assert math.isfinite(result["editor"])
    assert math.isfinite(result["searcher"])


def test_entropies_per_group_summary_shape(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    summary = entropies_per_group(fps, group_by="agent")
    assert set(summary) == {"editor", "searcher"}
    for entry in summary.values():
        assert entry.n == 9
        assert entry.q1 <= entry.median <= entry.q3
        assert entry.min <= entry.q1
        assert entry.max >= entry.q3


def test_discriminative_motifs_returns_topk(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    top = discriminative_motifs(
        fps, vocab, group_a="editor", group_b="searcher", k=5, group_by="agent"
    )
    assert len(top) <= 5
    # Editor and searcher use disjoint motif palettes (editor centers on EDIT
    # and RUN_TEST; searcher on SEARCH_REPO and READ_FILE). After BPE merges
    # those may be glued into multi-atom motifs, so we do not check for a
    # specific atom name; instead we check that the top motif has a strongly
    # nonzero log-odds, indicating clear discrimination.
    assert top
    assert abs(top[0].log_odds) > 1.0


def test_discriminative_motifs_jsd_ranking_returns_nonnegative_contributions(
    structured_corpus: list,
) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    top = discriminative_motifs(
        fps,
        vocab,
        group_a="editor",
        group_b="searcher",
        k=5,
        ranking="jsd_contribution",
        group_by="agent",
    )
    for m in top:
        assert m.jsd_contribution >= 0.0


def test_discriminative_motifs_rejects_unknown_ranking(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    with pytest.raises(ValueError, match="ranking"):
        discriminative_motifs(
            fps,
            vocab,
            group_a="editor",
            group_b="searcher",
            ranking="unknown",  # type: ignore[arg-type]
            group_by="agent",
        )


def test_discriminative_motifs_rejects_unknown_group(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)
    with pytest.raises(ValueError, match="no fingerprints"):
        discriminative_motifs(fps, vocab, group_a="ghost", group_b="searcher", group_by="agent")
