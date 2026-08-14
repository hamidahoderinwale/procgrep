"""Tests for `procgrep.io`."""

from __future__ import annotations

from pathlib import Path

from procgrep.bpe import fit_bpe
from procgrep.encode import encode
from procgrep.io import (
    fingerprints_to_records,
    read_jsonl,
    records_to_fingerprints,
    records_to_traces,
    traces_to_records,
    write_jsonl,
)


def test_trace_round_trip_through_jsonl(tmp_path: Path, small_corpus: list) -> None:
    path = tmp_path / "traces.jsonl"
    n_written = write_jsonl(path, traces_to_records(small_corpus))
    assert n_written == len(small_corpus)

    recovered = list(records_to_traces(read_jsonl(path)))
    assert [t.trace_id for t in recovered] == [t.trace_id for t in small_corpus]
    assert [t.atoms for t in recovered] == [t.atoms for t in small_corpus]


def test_fingerprint_round_trip_through_jsonl(tmp_path: Path, structured_corpus: list) -> None:
    sequences = [t.atoms for t in structured_corpus]
    vocab = fit_bpe(sequences, vocab_size=20)
    fps = encode(structured_corpus, vocab=vocab)

    path = tmp_path / "fp.jsonl"
    n_written = write_jsonl(path, fingerprints_to_records(fps))
    assert n_written == len(fps)

    recovered = list(records_to_fingerprints(read_jsonl(path)))
    for original, restored in zip(fps, recovered, strict=True):
        assert original.trace_id == restored.trace_id
        assert original.agent == restored.agent
        assert original.group == restored.group
        assert original.counts == restored.counts
        assert original.vocab_spec == restored.vocab_spec


def test_fingerprint_records_without_vocab_spec_load(tmp_path: Path) -> None:
    # Fingerprint files written before the spec existed have no vocab_spec key.
    path = tmp_path / "fp.jsonl"
    path.write_text('{"trace_id": "t", "agent": "a", "group": "g", "counts": [1, 2]}\n')
    (fp,) = records_to_fingerprints(read_jsonl(path))
    assert fp.vocab_spec is None
