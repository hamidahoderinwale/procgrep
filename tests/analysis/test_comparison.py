"""Comparison contracts shared by the Python API and CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from procgrep import Trace, encode, fit_bpe, jsd, jsd_matrix
from procgrep.cli import app
from procgrep.comparison import build_comparison


def _rows(*sequences: list[str]) -> list[dict[str, object]]:
    return [{"atoms_canonical": sequence} for sequence in sequences]


@pytest.mark.parametrize("base", [2.0, math.e])
def test_cli_and_python_api_write_identical_reports(tmp_path: Path, base: float) -> None:
    a = [{"atoms_canonical": ["read_file", "edit"], "resolved": True}] * 12
    b = [{"atoms_canonical": ["read_file", "run_test"], "resolved": False}] * 12
    paths = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for path, rows in zip(paths, (a, b), strict=True):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "comparison.json"
    result = CliRunner().invoke(
        app, ["compare", str(paths[0]), str(paths[1]), "-o", str(out), "--base", str(base)]
    )
    assert result.exit_code == 0, result.output
    report = build_comparison(a, b, name_a="a", name_b="b", base=base)
    assert json.loads(out.read_text()) == report.to_dict()
    assert report.summary() in result.output
    assert report.peak_step == 1
    assert report.peak_step_jsd == pytest.approx(math.log(2, base))
    assert report.a.n_labeled == report.a.n_resolved == 12
    assert report.b.n_resolved == 0


def test_identical_corpora_have_zero_divergence() -> None:
    rows = _rows(["read_file", "edit"]) * 12
    report = build_comparison(rows, rows)
    assert report.canonical_jsd == 0.0
    assert report.bigram_jsd == 0.0
    assert report.positional_curve == (0.0, 0.0)
    assert report.top_a == report.top_b == ()


def test_divergence_uses_shared_api_and_observed_actions() -> None:
    report = build_comparison(_rows(["submit", "submit"]), _rows(["localize", "localize"]))
    assert report.alphabet == ("localize", "submit")
    assert report.canonical_jsd == jsd([1, 0], [0, 1]) == 1.0
    assert report.bigram_jsd == 1.0
    assert report.top_a == (("submit|submit", 1.0),)
    assert report.top_b == (("localize|localize", 1.0),)
    assert report.to_dict()["smoothing"] == 0.0
    assert report.to_dict()["aggregation"] == "pooled_counts"
    assert report.to_dict()["schema_version"] == 2


def test_custom_actions_are_counted_without_extending_a_fixed_alphabet() -> None:
    report = build_comparison(_rows(["custom_action"]), _rows(["edit"]))
    assert report.canonical_jsd == 1.0
    assert "custom_action" in report.alphabet


def test_logarithm_base_is_explicit() -> None:
    report = build_comparison(_rows(["edit"]), _rows(["run_test"]), base=math.e)
    assert report.canonical_jsd == pytest.approx(math.log(2))
    assert report.to_dict()["base"] == math.e


def test_pooled_counts_are_distinct_from_equal_trace_weighting() -> None:
    a = _rows(["edit"] * 9, ["read_file"])
    b = _rows(["edit"], ["read_file"])
    report = build_comparison(a, b)
    assert report.canonical_jsd == pytest.approx(jsd([9, 1], [1, 1]))
    assert report.canonical_jsd > 0

    traces = [
        Trace(trace_id=f"{agent}-{i}", agent=agent, atoms=row["atoms_canonical"])
        for agent, rows in (("a", a), ("b", b))
        for i, row in enumerate(rows)
    ]
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=2)
    matrix = jsd_matrix(encode(traces, vocab=vocab), group_by="agent").to_array()
    assert matrix[0, 1] == 0.0


def test_record_order_does_not_change_the_report() -> None:
    a = _rows(["edit", "read_file"], ["edit", "run_test"]) * 6
    b = _rows(["read_file", "edit"], ["run_test", "edit"]) * 6
    assert build_comparison(a, b).to_dict() == build_comparison(a[::-1], b[::-1]).to_dict()


def test_absent_measurements_are_null_not_zero_or_nan() -> None:
    report = build_comparison(_rows([]), _rows(["edit"]))
    payload = report.to_dict()
    for field in ("canonical_jsd", "native_jsd", "bigram_jsd", "peak_step", "peak_step_jsd"):
        assert payload[field] is None
    assert report.positional_curve == ()
    assert "unavailable" in report.summary()
    json.dumps(payload, allow_nan=False)


def test_short_traces_have_no_bigram_measurement() -> None:
    report = build_comparison(_rows(["edit"]) * 12, _rows(["submit"]) * 12)
    assert report.canonical_jsd == 1.0
    assert report.bigram_jsd is None
    assert report.positional_curve == (1.0,)


def test_positions_require_coverage_on_both_sides() -> None:
    report = build_comparison(_rows(["edit"]) * 9, _rows(["submit"]) * 12)
    assert report.peak_step is None
    assert report.positional_curve == ()


def test_position_limit_and_top_n_are_respected() -> None:
    report = build_comparison(
        _rows(["edit", "edit"]) * 12,
        _rows(["submit", "submit"]) * 12,
        position_limit=1,
        top_n=0,
    )
    assert report.positional_curve == (1.0,)
    assert report.top_a == report.top_b == ()


@pytest.mark.parametrize("base", [0.0, 1.0, -1.0, math.nan, math.inf])
def test_invalid_bases_raise(base: float) -> None:
    with pytest.raises(ValueError, match="base"):
        build_comparison(_rows(["edit"]), _rows(["edit"]), base=base)


def test_empty_corpus_and_missing_schema_raise() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_comparison([], _rows(["edit"]))
    with pytest.raises(ValueError, match="atoms_canonical"):
        build_comparison([{"atoms": ["edit"]}], _rows(["edit"]))


def test_non_mapping_records_raise() -> None:
    with pytest.raises(ValueError, match="mappings"):
        build_comparison([None], _rows(["edit"]))  # type: ignore[list-item]


@pytest.mark.parametrize("actions", [None, "edit", [1]])
def test_malformed_action_sequences_raise(actions: object) -> None:
    with pytest.raises(ValueError, match="action names"):
        build_comparison([{"atoms_canonical": actions}], _rows(["edit"]))


def test_duplicate_names_and_invalid_limits_raise() -> None:
    rows = _rows(["edit"])
    with pytest.raises(ValueError, match="distinct"):
        build_comparison(rows, rows, name_a="same", name_b="same")
    with pytest.raises(ValueError, match="non-negative"):
        build_comparison(rows, rows, top_n=-1)
    with pytest.raises(ValueError, match="positive"):
        build_comparison(rows, rows, min_position_traces=0)
