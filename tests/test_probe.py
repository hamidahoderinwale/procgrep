"""Tests for `procgrep.probe`."""

from __future__ import annotations

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.probe import leave_one_group_out


def test_probe_recovers_agent_label_across_held_out_groups(
    structured_corpus: list,
) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)

    result = leave_one_group_out(fps, label_field="agent", seed=0)
    assert set(result.groups) == {"X", "Y", "Z"}
    # The two agents use disjoint procedures, so the agent label transfers
    # across held-out groups; overall accuracy should be near 1.0.
    assert result.overall_accuracy >= 0.9


def test_probe_confusion_records_all_predictions(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)

    result = leave_one_group_out(fps, label_field="agent", seed=0)
    for group in result.groups:
        total = sum(result.confusion[group].values())
        # Six trajectories per group: three from each agent.
        assert total == 6
