"""Contract tests for the `procgrep.program` programmability layer.

These pin the public contract a scaffold relies on, complementary to the
dispatch tests in `test_program.py`:

* ``enforce(mode="prompt")`` emits non-empty text that carries the spec's
  rules in prose.
* ``enforce(mode="guard")`` emits a `GuardArtifact` whose ``check`` flags a
  violating atom prefix and passes a clean one.
* ``verify`` reaches each of the three verdicts on constructed before/after
  populations.
* A derive -> enforce -> verify round-trip closes on synthetic traces.
* ``decode`` / ``reward`` modes and ``optimize`` are not implemented yet.
"""

from __future__ import annotations

from procgrep.bpe import ProcedureVocabulary, fit_bpe
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

TARGET_SHAPE = [ATOM_SEARCH_REPO, ATOM_READ_FILE, ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT]
OFF_SHAPE = [ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT, ATOM_EDIT]


def _vocab() -> ProcedureVocabulary:
    return fit_bpe([TARGET_SHAPE, OFF_SHAPE], vocab_size=20)


def _winners(n: int) -> list[Trace]:
    return [
        Trace(trace_id=f"w{i}", agent="a", atoms=TARGET_SHAPE, metadata={"resolved": True})
        for i in range(n)
    ]


def _losers(n: int) -> list[Trace]:
    return [
        Trace(trace_id=f"l{i}", agent="a", atoms=OFF_SHAPE, metadata={"resolved": False})
        for i in range(n)
    ]


def _pop(shape: list[str], n: int, *, resolved: bool, prefix: str) -> list[Trace]:
    return [
        Trace(trace_id=f"{prefix}{i}", agent="a", atoms=shape, metadata={"resolved": resolved})
        for i in range(n)
    ]


# --- enforce prompt contract ------------------------------------------------


def test_enforce_prompt_emits_nonempty_text_with_rules() -> None:
    # A spec with a named phase and a named penalty: both rules should be
    # visible in the rendered prompt prose, not just the boilerplate.
    spec = ProcedureSpec(
        phases=(Phase(name="explore", reward=0.3, require_any=(ATOM_SEARCH_REPO,)),),
        penalties=(Penalty(name="streak", reward=0.15, max_run=2),),
    )
    out = enforce(spec, mode="prompt")
    assert isinstance(out, str)
    assert out.strip()
    # The phase's required atom and the penalty cap surface in the prose.
    assert "search repo" in out
    assert "more than 2 edits in a row" in out
    # The prompt is the spec's own rendering.
    assert out == spec.to_prompt()


def test_enforce_default_mode_is_prompt() -> None:
    spec = ProcedureSpec(penalties=(Penalty(name="streak", reward=0.15, max_run=2),))
    assert enforce(spec) == spec.to_prompt()


# --- enforce guard contract -------------------------------------------------


def test_enforce_guard_check_flags_violation_and_passes_clean() -> None:
    spec = ProcedureSpec(penalties=(Penalty(name="streak", reward=0.15, max_run=2),))
    art = enforce(spec, mode="guard")
    assert isinstance(art, GuardArtifact)
    # Patterns mirror the spec's guard patterns.
    assert [p.name for p in art.patterns] == ["streak"]
    # A run of 3 edits exceeds the cap of 2: the streaming check flags it.
    assert art.check([ATOM_EDIT, ATOM_EDIT, ATOM_EDIT]) == ["streak"]
    # A clean prefix passes.
    assert art.check([ATOM_EDIT, ATOM_RUN_TEST, ATOM_EDIT]) == []


def test_enforce_guard_check_flags_forbidden_sequence() -> None:
    # A second guard kind: a forbidden contiguous sequence.
    spec = ProcedureSpec(
        penalties=(
            Penalty(name="blind_submit", reward=0.2, forbid_sequence=(ATOM_EDIT, ATOM_SUBMIT)),
        ),
    )
    art = enforce(spec, mode="guard")
    assert isinstance(art, GuardArtifact)
    assert art.check([ATOM_EDIT, ATOM_SUBMIT]) == ["blind_submit"]
    assert art.check([ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT]) == []


# --- verify verdicts --------------------------------------------------------


def test_verify_reaches_lever() -> None:
    vocab = _vocab()
    spec = ProcedureSpec.from_winners([*_winners(3), *_losers(1)], vocab, k=5)
    before = _pop(OFF_SHAPE, 4, resolved=False, prefix="b")
    after = _pop(TARGET_SHAPE, 4, resolved=True, prefix="a")
    report = verify(before, after, spec, vocab)
    assert isinstance(report, VerifyReport)
    assert report.verdict == "lever"
    assert report.behavior_moved
    assert report.outcome_delta > 0


def test_verify_reaches_epiphenomenal() -> None:
    vocab = _vocab()
    spec = ProcedureSpec.from_winners([*_winners(3), *_losers(1)], vocab, k=5)
    # Behavior moves toward the target, but the resolved rate does not budge.
    before = _pop(OFF_SHAPE, 4, resolved=False, prefix="b")
    after = _pop(TARGET_SHAPE, 4, resolved=False, prefix="a")
    report = verify(before, after, spec, vocab)
    assert report.verdict == "epiphenomenal"
    assert report.behavior_moved
    assert report.outcome_delta == 0.0


def test_verify_reaches_weak_enforcement() -> None:
    vocab = _vocab()
    spec = ProcedureSpec.from_winners([*_winners(3), *_losers(1)], vocab, k=5)
    # Both populations are off-target: enforcement did not take.
    before = _pop(OFF_SHAPE, 4, resolved=False, prefix="b")
    after = _pop(OFF_SHAPE, 4, resolved=True, prefix="a")
    report = verify(before, after, spec, vocab)
    assert report.verdict == "weak_enforcement"
    assert not report.behavior_moved


# --- derive -> enforce -> verify round-trip ---------------------------------


def test_derive_enforce_verify_round_trip() -> None:
    vocab = _vocab()
    corpus = [*_winners(4), *_losers(2)]

    # Derive a spec from the winners.
    spec = ProcedureSpec.from_winners(corpus, vocab, k=5)

    # Enforce as a prompt: a non-empty ruleset for a scaffold to inject.
    prompt = enforce(spec, mode="prompt")
    assert isinstance(prompt, str)
    assert prompt.strip()

    # Enforce as a guard: a usable artifact.
    art = enforce(spec, mode="guard")
    assert isinstance(art, GuardArtifact)

    # Verify the spec moved a synthetic population toward its own target and
    # closes the loop with a lever verdict.
    before = _pop(OFF_SHAPE, 4, resolved=False, prefix="b")
    after = _pop(TARGET_SHAPE, 4, resolved=True, prefix="a")
    report = verify(before, after, spec, vocab)
    assert report.verdict == "lever"
    before_jsd, after_jsd = report.fingerprint_jsd_to_target
    assert after_jsd < before_jsd


# --- not-yet-implemented modes ----------------------------------------------


def test_enforce_decode_returns_artifact() -> None:
    art = enforce(ProcedureSpec(), mode="decode")
    assert isinstance(art, DecodeArtifact)
    assert callable(art.allowed)
    assert art.alphabet


def test_enforce_reward_returns_artifact() -> None:
    art = enforce(ProcedureSpec(), mode="reward")
    assert isinstance(art, RewardArtifact)
    assert callable(art.reward)
    assert callable(art.step_rewards)


def test_optimize_returns_tuned_spec_and_report() -> None:
    spec = ProcedureSpec(phases=(Phase(name="verify", reward=0.5, require_any=(ATOM_RUN_TEST,)),))
    best, report = optimize(spec, [*_winners(3), *_losers(3)])
    assert isinstance(best, ProcedureSpec)
    assert report.best_val_score >= report.seed_val_score
