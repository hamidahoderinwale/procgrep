"""Tests for the one-shot corpus report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from procgrep.cli import app
from procgrep.report import build_report
from procgrep.types import PROCEDURE_SEPARATOR

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "examples" / "data" / "synthetic_traces.jsonl"


def test_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        build_report([])


def test_report_shape(structured_corpus) -> None:
    rep = build_report(structured_corpus, source="unit", vocab_size=24)
    assert rep.n_traces == len(structured_corpus)
    assert rep.source == "unit"
    assert 0.0 <= rep.exact_duplicate_rate <= 1.0
    assert abs(sum(share for _, share in rep.atom_mix) - 1.0) < 1e-9
    # two agents in the fixture: per-agent rows and at least one JSD pair
    assert len(rep.agents) == 2
    assert rep.jsd_pairs
    assert 0.0 <= rep.jsd_pairs[0][2] <= 1.0
    # multi-step procedures only
    assert all(PROCEDURE_SEPARATOR in proc for proc, _ in rep.top_procedures)


def test_summary_and_dict_round_trip(structured_corpus) -> None:
    rep = build_report(structured_corpus, vocab_size=24)
    text = rep.summary()
    assert "corpus" in text
    assert "action mix" in text
    payload = json.loads(json.dumps(rep.to_dict()))
    assert payload["n_traces"] == rep.n_traces


def test_cli_report_on_bundled_synthetic(tmp_path: Path) -> None:
    # the bundled file is raw, so canonicalize first, exactly as a user would
    canonical = tmp_path / "canonical.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "canonicalize",
            "--input",
            str(SYNTH),
            "--adapter",
            "swe-agent",
            "--output",
            str(canonical),
        ],
    )
    assert result.exit_code == 0, result.output
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        app, ["report", str(canonical), "--vocab-size", "24", "--json", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "corpus" in result.output
    assert json.loads(out.read_text())["n_traces"] > 0
