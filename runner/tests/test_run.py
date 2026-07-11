"""run_instance: prompt-artifact injection, task independence, failure stubs."""

from __future__ import annotations

import json

from helpers import READ_LOOP_SPEC, SUBMIT_COMMAND, local_env, scripted_config, scripted_model
from procgrep_runner.run import ARM_BASELINE, ARM_ENFORCED, run_instance, traj_path

INSTANCES = [
    {"instance_id": "demo__demo-1", "problem_statement": "fix the parser"},
    {"instance_id": "demo__demo-2", "problem_statement": "fix the cache"},
]


def run_one(tmp_path, instance, arm, run_name="run_v1"):
    return run_instance(
        instance,
        spec=READ_LOOP_SPEC,
        arm=arm,
        mode="prompt",
        config=scripted_config(),
        run_dir=tmp_path / run_name,
        model_factory=lambda config: scripted_model([("", [SUBMIT_COMMAND])]),
        env_factory=lambda config, inst: local_env(tmp_path),
    )


def system_message(tmp_path, instance, arm, run_name="run_v1"):
    path = traj_path(tmp_path / run_name, arm, instance["instance_id"], 0)
    return json.loads(path.read_text())["messages"][0]["content"]


def test_prompt_artifact_lands_only_in_the_enforced_arm(tmp_path):
    for instance in INSTANCES:
        for arm in (ARM_BASELINE, ARM_ENFORCED):
            row = run_one(tmp_path, instance, arm)
            assert row["exit_status"] == "Submitted"

    ruleset = READ_LOOP_SPEC.to_prompt()
    enforced = [system_message(tmp_path, i, ARM_ENFORCED) for i in INSTANCES]
    baseline = [system_message(tmp_path, i, ARM_BASELINE) for i in INSTANCES]
    assert all(ruleset in msg for msg in enforced)
    assert all(ruleset not in msg for msg in baseline)
    # No task leakage: the enforcement artifact is identical across tasks.
    assert enforced[0] == enforced[1]
    assert baseline[0] == baseline[1]


def test_env_failure_leaves_a_stub_trajectory_and_a_row(tmp_path):
    def broken_env(config, instance):
        raise RuntimeError("no sandbox")

    row = run_instance(
        INSTANCES[0],
        spec=READ_LOOP_SPEC,
        arm=ARM_BASELINE,
        mode="guard",
        config=scripted_config(),
        run_dir=tmp_path / "run_v1",
        model_factory=lambda config: scripted_model([]),
        env_factory=broken_env,
    )

    assert row["exit_status"] == "RuntimeError"
    assert row["submitted"] is False
    record = json.loads(traj_path(tmp_path / "run_v1", ARM_BASELINE, "demo__demo-1", 0).read_text())
    assert record["info"]["exit_status"] == "RuntimeError"
    assert record["messages"] == []
