"""Scripted-run helpers: a real mini-swe-agent loop with no API calls or docker.

The model is mini-swe-agent's own `DeterministicModel` (replays scripted
outputs); the environment is `LocalEnvironment` running harmless local
commands, so the Submitted path and observation schema are the real ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

from procgrep.reward import Penalty, ProcedureSpec
from procgrep.types import ATOM_READ_FILE

SYSTEM_TEMPLATE = "You are a scripted test agent."
INSTANCE_TEMPLATE = "Task: {{task}}"
SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo 'diff --git'"

READ_LOOP_SPEC = ProcedureSpec(
    name="no_read_loops",
    penalties=(Penalty(name="read_loop", reward=-1.0, forbid_sequence=(ATOM_READ_FILE,) * 3),),
)

AGENT_KWARGS = {
    "system_template": SYSTEM_TEMPLATE,
    "instance_template": INSTANCE_TEMPLATE,
    "cost_limit": 0,
    "step_limit": 0,
}


def scripted_model(turns: list[tuple[str, list[str]]]) -> DeterministicModel:
    """One output per turn: (assistant content, [bash commands])."""
    outputs = [
        make_output(content, [{"command": c} for c in commands]) for content, commands in turns
    ]
    return DeterministicModel(outputs=outputs)


def local_env(tmp_path: Path) -> LocalEnvironment:
    (tmp_path / "a.py").write_text("x = 1\n")
    return LocalEnvironment(cwd=str(tmp_path))


def scripted_config() -> dict[str, Any]:
    return {"agent": dict(AGENT_KWARGS), "model": {}, "environment": {}}


def write_traj(
    run_dir: Path,
    arm: str,
    instance_id: str,
    replicate: int,
    commands: list[str],
    *,
    exit_status: str = "Submitted",
) -> Path:
    """Handcraft a minimal mini-swe-agent trajectory file for measure tests."""
    record = {
        "instance_id": instance_id,
        "info": {"exit_status": exit_status, "submission": "diff --git"},
        "messages": [
            {"role": "assistant", "content": "", "extra": {"actions": [{"command": c}]}}
            for c in commands
        ],
    }
    path = run_dir / "arms" / arm / f"{instance_id}.r{replicate}.traj.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path
