"""Tests for `procgrep.reward` (procedural reward scoring)."""

from __future__ import annotations

from pathlib import Path

import pytest

from procgrep.reward import RewardResult, load_spec, score

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPEC = ROOT / "examples" / "rules" / "reward_spec_swe_agent.yaml"

INLINE_SPEC = """
name: test_spec
phases:
  - name: explore
    reward: 0.5
    require_any:
      - atom: read_file
penalties:
  - name: edit_streak
    pattern: [edit, edit, edit]
    contiguous: true
    penalty: 0.3
bonuses:
  - name: test_first
    require_sequence: [run_test]
    before_first: edit
    reward: 0.2
floor: 0.0
ceiling: 1.0
"""


@pytest.fixture
def inline_spec(tmp_path: Path):
    p = tmp_path / "spec.yaml"
    p.write_text(INLINE_SPEC)
    return load_spec(p)


# --- load_spec --------------------------------------------------------------


def test_load_example_spec_has_name_and_phases() -> None:
    spec = load_spec(EXAMPLE_SPEC)
    assert spec["name"]
    assert spec["phases"]  # at least one phase defined


def test_load_spec_missing_file_raises() -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        load_spec(Path("/nonexistent/spec.yaml"))


# --- score: structural guarantees -------------------------------------------


def test_score_returns_rewardresult_in_range(inline_spec) -> None:
    result = score(["read_file"], inline_spec)
    assert isinstance(result, RewardResult)
    assert 0.0 <= result.proc_score <= 1.0


def test_phase_satisfied_earns_reward(inline_spec) -> None:
    result = score(["read_file"], inline_spec)
    assert "explore" in result.satisfied_phases
    assert result.proc_score > 0.0


def test_penalty_fires_on_contiguous_edit_streak(inline_spec) -> None:
    result = score(["edit", "edit", "edit"], inline_spec)
    assert "edit_streak" in result.triggered_penalties
    # no phase satisfied + penalty -> clipped to floor
    assert result.proc_score == 0.0


def test_bonus_earned_when_test_before_first_edit(inline_spec) -> None:
    result = score(["run_test", "edit"], inline_spec)
    assert "test_first" in result.triggered_bonuses


def test_better_trajectory_scores_at_least_as_high(inline_spec) -> None:
    good = score(["read_file", "run_test", "edit"], inline_spec)
    bad = score(["edit", "edit", "edit"], inline_spec)
    assert good.proc_score >= bad.proc_score


def test_score_clipped_to_ceiling(inline_spec) -> None:
    # explore (0.5) + test_first (0.2) stays within the [0, 1] ceiling
    result = score(["read_file", "run_test", "edit"], inline_spec)
    assert result.proc_score <= 1.0


def test_empty_trajectory_scores_floor(inline_spec) -> None:
    result = score([], inline_spec)
    assert result.proc_score == 0.0
    assert result.satisfied_phases == []


def test_example_spec_scores_a_well_formed_trajectory() -> None:
    spec = load_spec(EXAMPLE_SPEC)
    atoms = ["search_repo", "read_file", "think", "edit", "run_test", "submit"]
    result = score(atoms, spec)
    assert isinstance(result, RewardResult)
    assert 0.0 <= result.proc_score <= 1.0
    assert result.satisfied_phases  # a structured run satisfies at least one phase
