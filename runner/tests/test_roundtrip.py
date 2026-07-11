"""Scripted actions -> trajectory file -> procgrep adapter -> the scripted atoms."""

from __future__ import annotations

import json

from helpers import AGENT_KWARGS, SUBMIT_COMMAND, local_env, scripted_model
from procgrep_runner.harness import GuardedAgent

from procgrep.ingest.adapters.mini_swe_agent import mini_swe_agent_adapter


def test_trajectory_roundtrips_to_scripted_atoms(tmp_path):
    turns = [
        ("inspect the file first", ["cat a.py"]),
        ("", ["grep -rn TODO ."]),
        ("", ["sed -i.bak s/x/y/ a.py"]),
        ("", ["pytest -x"]),
        ("", [SUBMIT_COMMAND]),
    ]
    agent = GuardedAgent(scripted_model(turns), local_env(tmp_path), **AGENT_KWARGS)
    info = agent.run("fix it")
    assert info["exit_status"] == "Submitted"

    traj_path = tmp_path / "run.traj.json"
    agent.save(traj_path, {"instance_id": "demo__demo-1"})
    record = json.loads(traj_path.read_text())

    # think from the first turn's non-empty content; submit echo classifies as other;
    # exit_status Submitted is normal, so no trailing error atom.
    assert mini_swe_agent_adapter(record) == [
        "think",
        "read_file",
        "search_repo",
        "edit",
        "run_test",
        "other",
    ]
