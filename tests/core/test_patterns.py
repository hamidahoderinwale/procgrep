"""Tests for `procgrep.patterns`."""

from __future__ import annotations

from pathlib import Path

import pytest

from procgrep.patterns import Pattern, load_patterns, match_patterns
from procgrep.types import (
    ATOM_EDIT,
    ATOM_LOCALIZE,
    ATOM_RUN_TEST,
    Trace,
)


def test_must_hold_true_passes_when_pattern_matches() -> None:
    trace = Trace(trace_id="t1", agent="a", atoms=[ATOM_LOCALIZE, ATOM_EDIT])
    pattern = Pattern(
        name="loc_before_edit",
        description="",
        pattern="localize .*edit",
        must_hold=True,
    )
    report = match_patterns([trace], [pattern])
    assert report.violations == {}
    assert report.pass_rate_per_rule["loc_before_edit"] == 1.0


def test_must_hold_true_fails_when_pattern_does_not_match() -> None:
    trace = Trace(trace_id="t1", agent="a", atoms=[ATOM_EDIT, ATOM_RUN_TEST])
    pattern = Pattern(
        name="loc_before_edit",
        description="",
        pattern="localize .*edit",
        must_hold=True,
    )
    report = match_patterns([trace], [pattern])
    assert report.violations == {"t1": ["loc_before_edit"]}
    assert report.pass_rate_per_rule["loc_before_edit"] == 0.0


def test_must_hold_false_fails_when_pattern_matches() -> None:
    trace = Trace(trace_id="t1", agent="a", atoms=[ATOM_EDIT] * 6)
    pattern = Pattern(
        name="no_long_edit_loops",
        description="",
        pattern=r"(edit ){5,}",
        must_hold=False,
    )
    report = match_patterns([trace], [pattern])
    assert report.violations == {"t1": ["no_long_edit_loops"]}


def test_must_hold_false_passes_when_pattern_does_not_match() -> None:
    trace = Trace(trace_id="t1", agent="a", atoms=[ATOM_EDIT, ATOM_RUN_TEST, ATOM_EDIT])
    pattern = Pattern(
        name="no_long_edit_loops",
        description="",
        pattern=r"(edit ){5,}",
        must_hold=False,
    )
    report = match_patterns([trace], [pattern])
    assert report.violations == {}


def test_load_patterns_parses_yaml(tmp_path: Path) -> None:
    text = """
rules:
  - name: r1
    description: rule one
    pattern: "edit run_test"
    must_hold: false
  - name: r2
    description: rule two
    pattern: "localize"
    must_hold: true
"""
    path = tmp_path / "rules.yaml"
    path.write_text(text)
    patterns = load_patterns(path)
    assert [p.name for p in patterns] == ["r1", "r2"]
    assert patterns[0].must_hold is False
    assert patterns[1].must_hold is True


def test_load_patterns_rejects_malformed_yaml(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("rules:\n  - just_a_string\n")
    with pytest.raises(TypeError, match="not a mapping"):
        load_patterns(path)


def test_load_patterns_requires_top_level_rules_key(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("not_rules: []\n")
    with pytest.raises(ValueError, match="rules"):
        load_patterns(path)


def test_pass_rate_handles_empty_trace_set() -> None:
    pattern = Pattern(name="r", description="", pattern="edit", must_hold=True)
    report = match_patterns([], [pattern])
    assert report.pass_rate_per_rule["r"] == 1.0
    assert report.violations == {}


def test_combined_pattern_set(structured_corpus: list) -> None:
    patterns = [
        Pattern(
            name="has_submit",
            description="every trajectory ends in submit",
            pattern="submit ",
            must_hold=True,
        ),
        Pattern(
            name="long_read_run",
            description="impossible: trajectory has 5+ consecutive reads",
            pattern=r"(read_file ){5,}",
            must_hold=False,
        ),
    ]
    report = match_patterns(structured_corpus, patterns)
    assert report.pass_rate_per_rule["has_submit"] == 1.0
    assert report.pass_rate_per_rule["long_read_run"] == 1.0
