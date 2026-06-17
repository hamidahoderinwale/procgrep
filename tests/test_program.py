"""Tests for `procgrep.program` (the enforce / verify / optimize loop).

Covers enforce mode dispatch and the model-free contract, plus verify's
behavior x outcome 2x2 reaching each of the three verdicts.
"""

from __future__ import annotations

import pytest

from procgrep.bpe import fit_bpe
from procgrep.encode import Fingerprint
from procgrep.patterns import Pattern
from procgrep.program import (
    DecodeArtifact,
    GuardArtifact,
    RewardArtifact,
    VerifyReport,
    enforce,
    optimize,
    verify,
)
from procgrep.reward import Penalty, Phase, ProcedureSpec
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


def test_enforce_decode_masks_edit_streak() -> None:
    spec = ProcedureSpec(penalties=(Penalty(name="streak", reward=0.5, max_run=2),))
    art = enforce(spec, mode="decode")
    assert isinstance(art, DecodeArtifact)
    # at the cap, another edit is masked out; other atoms remain allowed
    assert ATOM_EDIT not in art.allowed([ATOM_EDIT, ATOM_EDIT])
    assert ATOM_RUN_TEST in art.allowed([ATOM_EDIT, ATOM_EDIT])
    # below the cap, editing is still allowed
    assert ATOM_EDIT in art.allowed([ATOM_EDIT])


def test_enforce_decode_masks_forbidden_sequence() -> None:
    spec = ProcedureSpec(
        penalties=(Penalty(name="loop", reward=0.5, forbid_sequence=(ATOM_READ_FILE, ATOM_EDIT)),),
    )
    art = enforce(spec, mode="decode")
    assert isinstance(art, DecodeArtifact)
    # after read_file, edit would complete the forbidden 2-gram, so it is masked
    assert ATOM_EDIT not in art.allowed([ATOM_READ_FILE])
    assert ATOM_EDIT in art.allowed([ATOM_RUN_TEST])


def test_enforce_decode_never_empty_and_agrees_with_guard() -> None:
    spec = ProcedureSpec(penalties=(Penalty(name="streak", reward=0.5, max_run=2),))
    dec = enforce(spec, mode="decode")
    guard = enforce(spec, mode="guard")
    assert isinstance(dec, DecodeArtifact)
    assert isinstance(guard, GuardArtifact)
    # decode masks the edit exactly when one more edit would trip the guard
    assert ATOM_EDIT not in dec.allowed([ATOM_EDIT, ATOM_EDIT])
    assert guard.check([ATOM_EDIT, ATOM_EDIT, ATOM_EDIT]) == ["streak"]
    # the mask is never empty
    assert dec.allowed([ATOM_EDIT] * 10)


def _reward_spec() -> ProcedureSpec:
    return ProcedureSpec(
        phases=(Phase(name="verify", reward=0.5, require_any=(ATOM_RUN_TEST,)),),
        penalties=(Penalty(name="streak", reward=0.25, max_run=2),),
    )


def test_enforce_reward_full_matches_score() -> None:
    spec = _reward_spec()
    art = enforce(spec, mode="reward")
    assert isinstance(art, RewardArtifact)
    atoms = [ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT]
    assert art.reward(atoms) == spec.score(atoms).score


def test_enforce_reward_monotonic_and_penalized() -> None:
    art = enforce(_reward_spec(), mode="reward")
    assert isinstance(art, RewardArtifact)
    # satisfying the phase scores above not satisfying it
    assert art.reward([ATOM_RUN_TEST]) > art.reward([ATOM_EDIT])
    # a long edit streak deducts from an otherwise-equal trajectory
    clean = [ATOM_RUN_TEST]
    streaky = [ATOM_RUN_TEST, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT]
    assert art.reward(streaky) < art.reward(clean)


def test_enforce_reward_step_rewards_sum_to_full() -> None:
    art = enforce(_reward_spec(), mode="reward")
    assert isinstance(art, RewardArtifact)
    atoms = [ATOM_SEARCH_REPO, ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT]
    steps = art.step_rewards(atoms)
    assert len(steps) == len(atoms)
    # floor is 0 and the empty prefix scores 0, so increments sum to the full reward
    assert sum(steps) == pytest.approx(art.reward(atoms))


def test_enforce_reward_is_deterministic_and_serializes() -> None:
    import json

    art = enforce(_reward_spec(), mode="reward")
    assert isinstance(art, RewardArtifact)
    atoms = [ATOM_EDIT, ATOM_RUN_TEST]
    assert art.reward(atoms) == art.reward(atoms)
    payload = json.loads(art.spec_json)
    assert payload["floor"] == 0.0
    assert payload["ceiling"] == 1.0
    assert [p["name"] for p in payload["phases"]] == ["verify"]
    assert [p["name"] for p in payload["penalties"]] == ["streak"]


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


def test_optimize_tunes_cap_to_improve_discrimination() -> None:
    winners = _pop(TARGET_SHAPE, 6, True, "w")
    # losers also run a test, so the phase does not separate them; only the
    # edit-streak does, and only once the cap drops below their streak
    losers = _pop([ATOM_RUN_TEST, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT], 6, False, "l")
    spec = ProcedureSpec(
        phases=(Phase(name="verify", reward=0.5, require_any=(ATOM_RUN_TEST,)),),
        penalties=(Penalty(name="edit_streak", reward=0.3, max_run=10),),
    )
    best, report = optimize(spec, [*winners, *losers], seed=0)
    assert isinstance(best, ProcedureSpec)
    assert report.best_val_score >= report.seed_val_score
    assert report.n_candidates > 0
    # the lax seed cap of 10 is tuned down so the streak penalty bites
    cap = next(p.max_run for p in best.penalties if p.max_run is not None)
    assert cap is not None
    assert cap < 10


def test_optimize_requires_winners_and_losers() -> None:
    spec = ProcedureSpec(phases=(Phase(name="verify", reward=0.5, require_any=(ATOM_RUN_TEST,)),))
    with pytest.raises(ValueError, match="winners and losers"):
        optimize(spec, _pop(TARGET_SHAPE, 3, True, "w"))


def test_target_fingerprint_is_a_fingerprint() -> None:
    vocab = _vocab()
    spec = _spec_with_target(vocab)
    assert isinstance(spec.target, Fingerprint)
    assert spec.target.total > 0
