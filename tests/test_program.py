"""Tests for `procgrep.program` (the enforce / verify / optimize loop).

Covers enforce mode dispatch and the model-free contract, plus verify's
behavior x outcome 2x2 reaching each of the three verdicts.
"""

from __future__ import annotations

import pytest

from procgrep.bpe import fit_bpe
from procgrep.encode import Fingerprint
from procgrep.patterns import Pattern
from procgrep.program import GuardArtifact, VerifyReport, enforce, optimize, verify
from procgrep.reward import Penalty, ProcedureSpec
from procgrep.types import (
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    Trace,
)

# Atom shapes reused across the verify cases.
TARGET_SHAPE = [ATOM_SEARCH_REPO, ATOM_READ_FILE, ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT]
OFF_SHAPE = [ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT]


def _vocab() -> object:
    return fit_bpe([TARGET_SHAPE, OFF_SHAPE], vocab_size=20)


def _spec_with_target(vocab: object) -> ProcedureSpec:
    winners = [
        Trace(trace_id=f"w{i}", agent="a", atoms=TARGET_SHAPE, metadata={"resolved": True})
        for i in range(3)
    ]
    return ProcedureSpec.from_winners([*winners, _loser()], vocab, k=5)  # type: ignore[arg-type]


def _loser() -> Trace:
    return Trace(trace_id="seed_lose", agent="a", atoms=OFF_SHAPE, metadata={"resolved": False})


def _pop(shape: list[str], n: int, resolved: bool, prefix: str) -> list[Trace]:
    return [
        Trace(trace_id=f"{prefix}{i}", agent="a", atoms=shape, metadata={"resolved": resolved})
        for i in range(n)
    ]


# --- enforce ----------------------------------------------------------------


def test_enforce_prompt_returns_text() -> None:
    spec = ProcedureSpec(penalties=(Penalty(name="streak", reward=0.15, max_run=2),))
    out = enforce(spec, mode="prompt")
    assert isinstance(out, str)
    assert out == spec.to_prompt()


def test_enforce_guard_returns_patterns_and_check() -> None:
    spec = ProcedureSpec(penalties=(Penalty(name="streak", reward=0.15, max_run=2),))
    art = enforce(spec, mode="guard")
    assert isinstance(art, GuardArtifact)
    assert all(isinstance(p, Pattern) for p in art.patterns)
    # a run of 3 edits exceeds the cap of 2 and the streaming check flags it
    assert art.check([ATOM_EDIT, ATOM_EDIT, ATOM_EDIT]) == ["streak"]
    assert art.check([ATOM_EDIT, ATOM_RUN_TEST]) == []


def test_enforce_decode_not_implemented() -> None:
    spec = ProcedureSpec()
    with pytest.raises(NotImplementedError, match="decode"):
        enforce(spec, mode="decode")


def test_enforce_reward_not_implemented() -> None:
    spec = ProcedureSpec()
    with pytest.raises(NotImplementedError, match="reward"):
        enforce(spec, mode="reward")


def test_enforce_unknown_mode_raises() -> None:
    spec = ProcedureSpec()
    with pytest.raises(ValueError, match="unknown enforce mode"):
        enforce(spec, mode="bogus")  # type: ignore[arg-type]


# --- verify -----------------------------------------------------------------


def test_verify_requires_a_target() -> None:
    vocab = _vocab()
    spec = ProcedureSpec()  # no target
    with pytest.raises(ValueError, match=r"spec\.target"):
        verify([_loser()], [_loser()], spec, vocab)  # type: ignore[arg-type]


def test_verify_lever_when_behavior_and_outcome_move() -> None:
    vocab = _vocab()
    spec = _spec_with_target(vocab)
    # before: off-shape, unresolved. after: on-target shape, resolved.
    before = _pop(OFF_SHAPE, 4, resolved=False, prefix="b")
    after = _pop(TARGET_SHAPE, 4, resolved=True, prefix="a")
    report = verify(before, after, spec, vocab)  # type: ignore[arg-type]
    assert isinstance(report, VerifyReport)
    assert report.behavior_moved
    assert report.outcome_delta > 0
    assert report.verdict == "lever"


def test_verify_epiphenomenal_when_behavior_moves_but_outcome_flat() -> None:
    vocab = _vocab()
    spec = _spec_with_target(vocab)
    # behavior moves toward target, but outcome stays unresolved on both sides
    before = _pop(OFF_SHAPE, 4, resolved=False, prefix="b")
    after = _pop(TARGET_SHAPE, 4, resolved=False, prefix="a")
    report = verify(before, after, spec, vocab)  # type: ignore[arg-type]
    assert report.behavior_moved
    assert report.outcome_delta == 0.0
    assert report.verdict == "epiphenomenal"


def test_verify_weak_enforcement_when_behavior_does_not_move() -> None:
    vocab = _vocab()
    spec = _spec_with_target(vocab)
    # both populations are off-target: behavior did not move regardless of outcome
    before = _pop(OFF_SHAPE, 4, resolved=False, prefix="b")
    after = _pop(OFF_SHAPE, 4, resolved=True, prefix="a")
    report = verify(before, after, spec, vocab)  # type: ignore[arg-type]
    assert not report.behavior_moved
    assert report.verdict == "weak_enforcement"


def test_verify_jsd_pair_is_before_then_after() -> None:
    vocab = _vocab()
    spec = _spec_with_target(vocab)
    before = _pop(OFF_SHAPE, 3, resolved=False, prefix="b")
    after = _pop(TARGET_SHAPE, 3, resolved=True, prefix="a")
    report = verify(before, after, spec, vocab)  # type: ignore[arg-type]
    before_jsd, after_jsd = report.fingerprint_jsd_to_target
    assert after_jsd < before_jsd  # moved closer to target


# --- optimize ---------------------------------------------------------------


def test_optimize_not_implemented() -> None:
    vocab = _vocab()
    spec = ProcedureSpec()
    with pytest.raises(NotImplementedError, match="roadmap"):
        optimize(spec, [_loser()], vocab)  # type: ignore[arg-type]


def test_target_fingerprint_is_a_fingerprint() -> None:
    vocab = _vocab()
    spec = _spec_with_target(vocab)
    assert isinstance(spec.target, Fingerprint)
    assert spec.target.total > 0
