"""Guard enforcement plumbing: block, steer, and the guard record in the trajectory."""

from __future__ import annotations

from helpers import AGENT_KWARGS, READ_LOOP_SPEC, SUBMIT_COMMAND, local_env, scripted_model
from procgrep_runner.harness import GuardedAgent

from procgrep.guard import ProcedureGuard


def make_agent(tmp_path, turns, on_violation):
    guard = ProcedureGuard(READ_LOOP_SPEC, on_violation=on_violation)
    return GuardedAgent(scripted_model(turns), local_env(tmp_path), guard=guard, **AGENT_KWARGS)


READ_LOOP_TURNS = [
    ("look around", ["cat a.py"]),
    ("", ["cat a.py"]),
    ("", ["cat a.py"]),  # third consecutive read: the spec forbids this
    ("", [SUBMIT_COMMAND]),
]


def test_block_substitutes_notice_and_does_not_commit(tmp_path):
    agent = make_agent(tmp_path, READ_LOOP_TURNS, "block")
    info = agent.run("fix it")

    assert info["exit_status"] == "Submitted"
    assert info["submission"].strip() == "diff --git"
    blocked = [e for e in agent.guard_events if not e["allowed"]]
    assert [e["action_index"] for e in blocked] == [2]
    assert blocked[0]["directive"] == "block"
    # The blocked read was never committed: two reads, then the submit echo.
    assert agent.guard.prefix == ("read_file", "read_file", "other")
    # The model saw a real observation carrying the block notice.
    assert any("[procedure-guard]" in str(m.get("content", "")) for m in agent.messages)


def test_block_recorded_in_serialized_trajectory(tmp_path):
    agent = make_agent(tmp_path, READ_LOOP_TURNS, "block")
    agent.run("fix it")
    guard_record = agent.serialize()["procgrep_runner"]["guard"]

    assert guard_record["spec"] == "no_read_loops"
    assert guard_record["on_violation"] == "block"
    assert guard_record["blocked"] == 1
    assert guard_record["steered"] == 0
    assert guard_record["checks"] == 4


def test_steer_runs_action_and_injects_rule_text(tmp_path):
    agent = make_agent(tmp_path, READ_LOOP_TURNS, "steer")
    agent.run("fix it")

    steers = [m for m in agent.messages if (m.get("extra") or {}).get("procgrep_guard_steer")]
    assert len(steers) == 1
    assert steers[0]["content"] == READ_LOOP_SPEC.to_prompt()
    # Steered actions still run, so all three reads were committed.
    assert agent.guard.prefix == ("read_file", "read_file", "read_file", "other")
    serialized = agent.serialize()["procgrep_runner"]["guard"]
    assert serialized["steered"] == 1
    assert serialized["blocked"] == 0


def test_no_guard_behaves_like_default_agent(tmp_path):
    agent = GuardedAgent(
        scripted_model(READ_LOOP_TURNS), local_env(tmp_path), guard=None, **AGENT_KWARGS
    )
    info = agent.run("fix it")

    assert info["exit_status"] == "Submitted"
    assert agent.guard_events == []
    assert agent.serialize()["procgrep_runner"] == {"action_count": 4}
