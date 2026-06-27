"""Tests for `procgrep.ingest.adapters.swe_agent_traj`.

SWE-bench leaderboard ``.traj`` submissions wrap a ``trajectory`` list whose
steps carry a full command string in ``action`` (not a clean action name), so
the adapter classifies the command: ``str_replace_editor`` operations map by
their second token, everything else routes through the shared terminal
classifier and onto the canonical alphabet. A non-empty ``thought`` prepends
``think``. Covers editor ops, shell-command classification, submit, the
thought prefix, and registration.
"""

from __future__ import annotations

from procgrep.canonicalize import get_adapter
from procgrep.ingest.adapters.swe_agent_traj import swe_agent_traj_adapter
from procgrep.types import (
    ATOM_CREATE_FILE,
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_CODE,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    ATOM_THINK,
)


def _traj(*steps: dict) -> dict:
    return {"trajectory": list(steps)}


def test_editor_ops_map_by_second_token() -> None:
    rec = _traj(
        {"action": "str_replace_editor view /repo/a.py", "thought": ""},
        {"action": "str_replace_editor str_replace /repo/a.py", "thought": ""},
        {"action": "str_replace_editor create /repo/b.py", "thought": ""},
    )
    assert swe_agent_traj_adapter(rec) == [ATOM_READ_FILE, ATOM_EDIT, ATOM_CREATE_FILE]


def test_shell_commands_route_through_terminal_classifier() -> None:
    rec = _traj(
        {"action": "find /repo -name '*.py' | grep model", "thought": ""},
        {"action": "cat /repo/setup.py", "thought": ""},
        {"action": "python repro.py", "thought": ""},
    )
    assert swe_agent_traj_adapter(rec) == [ATOM_SEARCH_REPO, ATOM_READ_FILE, ATOM_RUN_CODE]


def test_thought_prepends_think_and_submit() -> None:
    rec = _traj(
        {"action": "str_replace_editor view /repo/a.py", "thought": "let me look"},
        {"action": "submit", "thought": ""},
    )
    assert swe_agent_traj_adapter(rec) == [ATOM_THINK, ATOM_READ_FILE, ATOM_SUBMIT]


def test_empty_and_malformed_steps_are_skipped() -> None:
    rec = _traj({"thought": "no action here"}, "not-a-dict", {"action": "   "})
    # Only the lone thought yields an atom; the others contribute nothing.
    assert swe_agent_traj_adapter(rec) == [ATOM_THINK]


def test_registered_under_name() -> None:
    assert get_adapter("swe-agent-traj") is swe_agent_traj_adapter
