"""Tests for `procgrep.probe`."""

from __future__ import annotations

import pytest

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.probe import leave_one_group_out
from procgrep.types import ATOM_EDIT, ATOM_RUN_TEST, Trace


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


def test_probe_raises_on_single_group() -> None:
    # A single group cannot be held out; the probe must fail with a
    # domain-specific message rather than an opaque sklearn error.
    traces = [
        Trace(trace_id=f"t{i}", agent="editor", group="X", atoms=[ATOM_EDIT, ATOM_RUN_TEST])
        for i in range(4)
    ]
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=10)
    fps = encode(traces, vocab=vocab)
    with pytest.raises(ValueError, match="2 groups"):
        leave_one_group_out(fps, label_field="agent", seed=0)


def test_probe_confusion_records_all_predictions(structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)

    result = leave_one_group_out(fps, label_field="agent", seed=0)
    for group in result.groups:
        total = sum(result.confusion[group].values())
        # Six trajectories per group: three from each agent.
        assert total == 6
