"""Tests for the typed `ProcedureSpec` API in `procgrep.reward`.

Covers deriving a spec from winners, load-time YAML validation (a typo
raises), scoring with both clipped and raw totals, the prompt rendering,
and the guard-pattern rendering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from procgrep.bpe import fit_bpe
from procgrep.patterns import Pattern, match_patterns
from procgrep.reward import Penalty, Phase, ProcedureSpec, RewardResult
from procgrep.types import (
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    Trace,
)

pytest.importorskip("yaml")


def _winner_corpus() -> list[Trace]:
    """Winners explore-then-test, losers edit-streak with no test."""
    traces: list[Trace] = []
    for i in range(4):
        traces.append(
            Trace(
                trace_id=f"win{i}",
                agent="a",
                atoms=[
                    ATOM_SEARCH_REPO,
                    ATOM_READ_FILE,
                    ATOM_EDIT,
                    ATOM_RUN_TEST,
                    ATOM_SUBMIT,
                ],
                metadata={"resolved": True},
            )
        )
    for i in range(4):
        traces.append(
            Trace(
                trace_id=f"lose{i}",
                agent="a",
                atoms=[ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT],
                metadata={"resolved": False},
            )
        )
    return traces


# --- from_winners -----------------------------------------------------------


def test_from_winners_builds_phases_and_target() -> None:
    traces = _winner_corpus()
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=20)
    spec = ProcedureSpec.from_winners(traces, vocab, k=5)

    assert spec.phases  # derived at least one phase
    assert spec.target is not None
    # winners explore before edit and test after edit -> those phases present
    names = {p.name for p in spec.phases}
    assert "verification" in names
    assert "exploration" in names


def test_from_winners_caps_edit_streak_at_passing_distribution() -> None:
    traces = _winner_corpus()
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=20)
    spec = ProcedureSpec.from_winners(traces, vocab, k=5)

    streak = next((p for p in spec.penalties if p.name == "edit_streak"), None)
    assert streak is not None
    # winners never streak edits, so the percentile cap is clamped to its floor
    # of 3; the 5-edit loser still trips it
    assert streak.max_run == 3
    loser_atoms = [ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT]
    assert "edit_streak" in spec.score(loser_atoms).triggered_penalties


def test_from_winners_no_winners_raises() -> None:
    traces = [Trace(trace_id="x", agent="a", atoms=[ATOM_EDIT], metadata={"resolved": False})]
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=5)
    with pytest.raises(ValueError, match="no winners"):
        ProcedureSpec.from_winners(traces, vocab)


def test_derived_spec_scores_winner_above_loser() -> None:
    traces = _winner_corpus()
    vocab = fit_bpe([t.atoms for t in traces], vocab_size=20)
    spec = ProcedureSpec.from_winners(traces, vocab, k=5)

    win = spec.score([ATOM_SEARCH_REPO, ATOM_READ_FILE, ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT])
    lose = spec.score([ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT])
    assert win.score > lose.score


# --- from_yaml validation ---------------------------------------------------

GOOD_SPEC = """
name: good
phases:
  - name: explore
    reward: 0.3
    require_any: [search_repo, read_file]
    before_first: edit
penalties:
  - name: edit_streak
    reward: 0.15
    max_run: 3
floor: 0.0
ceiling: 1.0
"""

TYPO_SPEC = """
name: typo
phases:
  - name: explore
    reward: 0.3
    require_any: [read_file]
    min_counts: 2
"""


def test_from_yaml_loads_valid_spec(tmp_path: Path) -> None:
    p = tmp_path / "good.yaml"
    p.write_text(GOOD_SPEC)
    spec = ProcedureSpec.from_yaml(p)
    assert spec.name == "good"
    assert len(spec.phases) == 1
    assert spec.phases[0].before_first == ATOM_EDIT
    assert spec.penalties[0].max_run == 3


def test_from_yaml_typo_raises_at_load_time(tmp_path: Path) -> None:
    p = tmp_path / "typo.yaml"
    p.write_text(TYPO_SPEC)
    with pytest.raises(ValueError, match="unrecognized keys"):
        ProcedureSpec.from_yaml(p)


def test_from_yaml_missing_required_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "missing.yaml"
    p.write_text("phases:\n  - name: explore\n")  # no reward
    with pytest.raises(ValueError, match="missing required key 'reward'"):
        ProcedureSpec.from_yaml(p)


# --- score: raw and clipped -------------------------------------------------


def _scoring_spec() -> ProcedureSpec:
    return ProcedureSpec(
        phases=(
            Phase(name="explore", reward=0.6, require_any=(ATOM_READ_FILE,)),
            Phase(name="verify", reward=0.6, require_any=(ATOM_RUN_TEST,)),
        ),
        penalties=(Penalty(name="streak", reward=0.5, max_run=2),),
        floor=0.0,
        ceiling=1.0,
    )


def test_score_returns_raw_and_clipped() -> None:
    spec = _scoring_spec()
    result = spec.score([ATOM_READ_FILE, ATOM_RUN_TEST])
    assert isinstance(result, RewardResult)
    # 0.6 + 0.6 = 1.2 raw, clipped to ceiling 1.0
    assert result.raw == pytest.approx(1.2)
    assert result.score == pytest.approx(1.0)
    assert result.score <= 1.0 < result.raw


def test_score_raw_can_go_negative_below_floor() -> None:
    spec = _scoring_spec()
    # 4 edits trips the max_run=2 penalty (-0.5), no phase satisfied
    result = spec.score([ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT])
    assert "streak" in result.triggered_penalties
    assert result.raw == pytest.approx(-0.5)
    assert result.score == 0.0  # clipped to floor


def test_score_accepts_a_trace() -> None:
    spec = _scoring_spec()
    trace = Trace(trace_id="t", agent="a", atoms=[ATOM_READ_FILE])
    result = spec.score(trace)
    assert "explore" in result.satisfied_phases


def test_phase_before_first_scopes_the_prefix() -> None:
    spec = ProcedureSpec(
        phases=(
            Phase(
                name="explore_first",
                reward=0.5,
                require_any=(ATOM_READ_FILE,),
                before_first=ATOM_EDIT,
            ),
        )
    )
    # read after the first edit does not count
    assert spec.score([ATOM_EDIT, ATOM_READ_FILE]).score == 0.0
    assert spec.score([ATOM_READ_FILE, ATOM_EDIT]).score == pytest.approx(0.5)


def test_forbid_sequence_penalty_fires_on_contiguous_match() -> None:
    spec = ProcedureSpec(
        penalties=(Penalty(name="bad", reward=0.4, forbid_sequence=(ATOM_EDIT, ATOM_SUBMIT)),),
    )
    assert "bad" in spec.score([ATOM_EDIT, ATOM_SUBMIT]).triggered_penalties
    assert "bad" not in spec.score([ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT]).triggered_penalties


# --- to_prompt --------------------------------------------------------------


def test_to_prompt_renders_phases_and_penalties() -> None:
    spec = _scoring_spec()
    prompt = spec.to_prompt()
    assert isinstance(prompt, str)
    assert "read file" in prompt  # humanized atom
    assert "run test" in prompt
    assert "Avoid:" in prompt
    assert "in a row" in prompt  # max_run rendering


# --- to_patterns ------------------------------------------------------------


def test_to_patterns_renders_guard_patterns() -> None:
    spec = ProcedureSpec(
        penalties=(
            Penalty(name="streak", reward=0.15, max_run=2),
            Penalty(name="forbidden", reward=0.2, forbid_sequence=(ATOM_EDIT, ATOM_SUBMIT)),
        ),
    )
    patterns = spec.to_patterns()
    assert all(isinstance(p, Pattern) for p in patterns)
    assert {p.name for p in patterns} == {"streak", "forbidden"}
    assert all(p.must_hold is False for p in patterns)


def test_to_patterns_catches_the_offending_trace() -> None:
    spec = ProcedureSpec(penalties=(Penalty(name="streak", reward=0.15, max_run=2),))
    patterns = spec.to_patterns()
    offender = Trace(
        trace_id="bad",
        agent="a",
        atoms=[ATOM_EDIT, ATOM_EDIT, ATOM_EDIT],  # run of 3 > cap 2
    )
    clean = Trace(trace_id="ok", agent="a", atoms=[ATOM_EDIT, ATOM_RUN_TEST, ATOM_EDIT])
    report = match_patterns([offender, clean], patterns)
    assert "bad" in report.violations
    assert "ok" not in report.violations
