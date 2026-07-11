"""The measurement path: deterministic, direction-correct, uncertainty-quantified."""

from __future__ import annotations

import pytest
from helpers import READ_LOOP_SPEC, write_traj
from procgrep_runner.measure import measure_run

from procgrep.types import Trace

READ_LOOP = ["cat a.py"] * 6
# read -> edit -> test: atom-for-atom the winners' procedure below, so the
# enforced arm and the target share BPE segments (see measure design note 4).
EDIT_TEST = ["cat a.py", "sed -i s/x/y/ a.py", "pytest -x"]
IDS = ("demo__demo-1", "demo__demo-2", "demo__demo-3")


def winner_corpus() -> list[Trace]:
    winners = [
        Trace(f"w{i}", "winner", ["read_file", "edit", "run_test"], metadata={"resolved": True})
        for i in range(4)
    ]
    losers = [
        Trace(f"l{i}", "loser", ["read_file"] * 8, metadata={"resolved": False}) for i in range(4)
    ]
    return winners + losers


@pytest.fixture
def run_dir(tmp_path):
    run_dir = tmp_path / "run_v1"
    run_dir.mkdir()
    READ_LOOP_SPEC.to_yaml(run_dir / "spec.yaml")
    for iid in IDS:
        write_traj(run_dir, "baseline", iid, 0, READ_LOOP)
        write_traj(run_dir, "enforced", iid, 0, EDIT_TEST)
    return run_dir


def test_behavior_axis_moves_toward_winner_target(run_dir):
    summary = measure_run(run_dir, winners=winner_corpus(), n_boot=50)

    assert summary["verify"]["behavior_moved"] is True
    before_jsd, after_jsd = summary["verify"]["fingerprint_jsd_to_target"]
    assert after_jsd < before_jsd
    # No grades: the outcome axis is flat, so a behavior move reads epiphenomenal.
    assert summary["graded"] is False
    assert summary["verify"]["verdict"] == "epiphenomenal"
    assert summary["n"] == {"before": 3, "after": 3}
    assert (run_dir / "measure.summary.json").exists()
    assert (run_dir / "measure.rows.jsonl").read_text().count("\n") == 6


def test_grades_flip_the_verdict_to_lever(run_dir):
    grades = {"baseline": dict.fromkeys(IDS, False), "enforced": dict.fromkeys(IDS, True)}
    summary = measure_run(run_dir, winners=winner_corpus(), grades=grades, n_boot=50)

    assert summary["graded"] is True
    assert summary["verify"]["outcome_delta"] == 1.0
    assert summary["verify"]["verdict"] == "lever"


def test_measurement_is_deterministic(run_dir):
    first = measure_run(run_dir, winners=winner_corpus(), n_boot=100, write=False)
    second = measure_run(run_dir, winners=winner_corpus(), n_boot=100, write=False)

    assert first == second


def test_bootstrap_ci_excludes_zero_for_a_real_move(run_dir):
    summary = measure_run(run_dir, winners=winner_corpus(), n_boot=200, write=False)
    ci = summary["bootstrap"]

    assert ci["n_paired_instances"] == 3
    lo, hi = ci["jsd_move_toward_target_ci95"]
    assert 0 < lo <= hi


def test_target_required_when_spec_has_none(run_dir):
    with pytest.raises(ValueError, match="no target fingerprint"):
        measure_run(run_dir, write=False)
