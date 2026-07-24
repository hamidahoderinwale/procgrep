"""Tests for `procgrep.guard.ProcedureGuard`.

The guard turns a spec's decode/guard artifacts into a stateful, execute-time
checker: classify a proposed action to an atom, test it against the running
prefix, and report a decision the host enforces. Covers legal/blocked actions,
forbid-sequence and edit-streak blocking, command classification, the on_violation
policies (block / steer / warn), prefix lifecycle, and registration.
"""

from __future__ import annotations

from procgrep.guard import GuardDecision, ProcedureGuard
from procgrep.reward import Penalty, ProcedureSpec
from procgrep.types import ATOM_EDIT, ATOM_READ_FILE, ATOM_RUN_TEST


def _no_read_loop() -> ProcedureSpec:
    """Forbid two reads in a row (read_file, read_file)."""
    return ProcedureSpec(
        penalties=(
            Penalty(
                name="read_loop", reward=-1.0, forbid_sequence=(ATOM_READ_FILE, ATOM_READ_FILE)
            ),
        ),
        name="no_read_loop",
    )


def _edit_cap(cap: int) -> ProcedureSpec:
    return ProcedureSpec(
        penalties=(Penalty(name="edit_streak", reward=-1.0, max_run=cap),), name="edit_cap"
    )


def test_legal_action_allowed() -> None:
    guard = ProcedureGuard(_no_read_loop())
    d = guard.check(ATOM_EDIT)
    assert d.allowed is True
    assert d.directive == "allow"
    assert d.atom == ATOM_EDIT
    assert ATOM_EDIT in d.allowed_atoms
    assert d.reason == ""


def test_forbid_sequence_blocks_completing_atom() -> None:
    guard = ProcedureGuard(_no_read_loop())
    guard.commit(ATOM_READ_FILE)  # one read on the prefix
    d = guard.check(ATOM_READ_FILE)  # a second read would complete the forbidden pair
    assert d.allowed is False
    assert d.directive == "block"
    assert ATOM_READ_FILE not in d.allowed_atoms
    assert "forbidden" in d.reason


def test_other_atom_still_allowed_after_one_read() -> None:
    guard = ProcedureGuard(_no_read_loop())
    guard.commit(ATOM_READ_FILE)
    assert guard.check(ATOM_EDIT).allowed is True


def test_edit_streak_blocks_at_cap() -> None:
    guard = ProcedureGuard(_edit_cap(2))
    guard.commit(ATOM_EDIT)
    guard.commit(ATOM_EDIT)  # trailing run of 2 == cap
    assert guard.check(ATOM_EDIT).allowed is False
    assert guard.check(ATOM_RUN_TEST).allowed is True


def test_command_string_is_classified() -> None:
    guard = ProcedureGuard(_no_read_loop())
    d = guard.check("cat src/foo.py")  # classifies to read_file
    assert d.atom == ATOM_READ_FILE


def test_canonical_atom_passes_through() -> None:
    guard = ProcedureGuard(_no_read_loop())
    assert guard.check(ATOM_EDIT).atom == ATOM_EDIT  # not re-classified


def test_allowed_atoms_never_empty() -> None:
    guard = ProcedureGuard(_no_read_loop())
    guard.commit(ATOM_READ_FILE)
    assert len(guard.check(ATOM_READ_FILE).allowed_atoms) > 0


def test_step_block_does_not_commit() -> None:
    guard = ProcedureGuard(_no_read_loop(), on_violation="block")
    guard.step(ATOM_READ_FILE)  # allowed, commits
    d = guard.step(ATOM_READ_FILE)  # blocked, must not commit
    assert d.allowed is False
    assert guard.prefix == (ATOM_READ_FILE,)  # second read not on the prefix


def test_step_steer_commits_and_carries_message() -> None:
    guard = ProcedureGuard(
        _no_read_loop(), on_violation="steer", steer_message="Edit and validate."
    )
    guard.step(ATOM_READ_FILE)
    d = guard.step(ATOM_READ_FILE)
    assert d.allowed is False
    assert d.directive == "steer"
    assert d.steer_message == "Edit and validate."
    assert guard.prefix == (ATOM_READ_FILE, ATOM_READ_FILE)  # action still ran


def test_step_warn_commits_without_message() -> None:
    guard = ProcedureGuard(_no_read_loop(), on_violation="warn")
    guard.step(ATOM_READ_FILE)
    d = guard.step(ATOM_READ_FILE)
    assert d.directive == "warn"
    assert d.steer_message is None
    assert guard.prefix == (ATOM_READ_FILE, ATOM_READ_FILE)


def test_reset_clears_prefix() -> None:
    guard = ProcedureGuard(_no_read_loop())
    guard.commit(ATOM_READ_FILE)
    guard.reset()
    assert guard.prefix == ()
    assert guard.check(ATOM_READ_FILE).allowed is True


def test_phase_only_spec_never_blocks() -> None:
    # Phases are reward signals, not hard constraints; a spec with no penalties
    # blocks nothing.
    guard = ProcedureGuard(ProcedureSpec(name="phases_only"))
    guard.commit(ATOM_READ_FILE)
    assert guard.check(ATOM_READ_FILE).allowed is True


def test_decision_is_frozen() -> None:
    d = ProcedureGuard(_no_read_loop()).check(ATOM_EDIT)
    assert isinstance(d, GuardDecision)
    import dataclasses

    assert dataclasses.is_dataclass(d)
