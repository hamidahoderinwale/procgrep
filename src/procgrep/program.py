"""The programmability loop: enforce, verify, optimize a `ProcedureSpec`.

procgrep is model-free: there is no model in the loop, which is the whole point.
So the loop here does not run, wrap, or call an agent. It works on traces and
specs only:

* `enforce` EMITS the rendered enforcement artifact (a system prompt, or guard
  patterns plus a streaming check) for an external scaffold to apply. It does
  not run anything.
* `verify` reads the before/after trace populations a scaffold produced and
  reports the behavior x outcome 2x2: did procedure move toward the target, and
  did the resolved rate move.
* `optimize` is the search over specs; stubbed with a roadmap note.

Design decisions:

* `enforce` returns an artifact, never a running agent. Benefit: keeps procgrep
  a post-hoc, model-free library; the scaffold owns the model. Price: the caller
  must wire the artifact into their own runner.
* `verify` separates behavior from outcome on purpose. Benefit: it can name the
  failure where a spec changed behavior but not results ("epiphenomenal") versus
  changed nothing ("weak_enforcement") versus worked ("lever"). Price: it needs
  both a target fingerprint and an outcome label, so a spec with no target
  cannot be verified on the behavior axis.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, overload

import numpy as np

from procgrep.bpe import ProcedureVocabulary
from procgrep.encode import encode
from procgrep.jsd import jsd
from procgrep.patterns import Pattern
from procgrep.reward import ProcedureSpec
from procgrep.scaffolds import to_openhands_skill, to_swe_agent_config
from procgrep.types import (
    ATOM_EDIT,
    CANONICAL_ATOMS,
    Trace,
)

EnforceMode = Literal["prompt", "guard", "decode", "reward"]
Scaffold = Literal["swe-agent", "openhands"]
Verdict = Literal["lever", "epiphenomenal", "weak_enforcement"]


@dataclass(frozen=True)
class GuardArtifact:
    """The guard-mode enforcement artifact.

    ``patterns`` are the rendered guard patterns; ``check`` is a
    streaming callable a scaffold can apply to a growing atom prefix to
    decide whether the trajectory has already violated a guard. The
    callable is pure and holds no model.
    """

    patterns: tuple[Pattern, ...]
    check: Callable[[Sequence[str]], list[str]]


@dataclass(frozen=True)
class RewardArtifact:
    """The reward-mode enforcement artifact: a deterministic process reward.

    ``reward`` scores a whole trajectory in ``[floor, ceiling]`` (= ``spec.score``).
    ``step_rewards`` returns the per-step increments as the trajectory grows;
    they sum to ``reward`` of the full trajectory minus ``reward`` of the empty
    prefix, so a trainer can use them as a dense shaping signal instead of a
    single terminal reward. ``spec_json`` serializes the scoring rules so the
    reward can be reloaded without importing procgrep in the training hot loop.
    No model is in the loop.
    """

    reward: Callable[[Sequence[str]], float]
    step_rewards: Callable[[Sequence[str]], list[float]]
    spec_json: str


@dataclass(frozen=True)
class DecodeArtifact:
    """The decode-mode enforcement artifact: a constraint over the action grammar.

    ``alphabet`` is the canonical atom vocabulary a decoder chooses from.
    ``allowed`` maps an atom prefix to the atoms that may legally follow it
    without committing a penalty the spec forbids — an edit past the streak
    cap, or the final atom of a forbidden sequence. It always returns a
    non-empty set, so a constrained decoder is never wedged. ``rules_json``
    serializes the forbidding rules so an external sampler (a logit mask,
    outlines, llguidance) can apply the same constraint without procgrep in
    the loop. No model is in the loop.
    """

    alphabet: tuple[str, ...]
    allowed: Callable[[Sequence[str]], set[str]]
    rules_json: str


@dataclass(frozen=True)
class VerifyReport:
    """The behavior x outcome 2x2 for one enforcement attempt.

    Attributes:
        behavior_moved: True when the after-population's procedure
            distribution moved measurably toward ``spec.target``.
        fingerprint_jsd_to_target: ``(before_jsd, after_jsd)`` from each
            population mean to the target. Lower is closer.
        outcome_delta: After resolved-rate minus before resolved-rate.
        verdict: ``"lever"`` (behavior moved and outcome improved),
            ``"epiphenomenal"`` (behavior moved, outcome did not), or
            ``"weak_enforcement"`` (behavior did not move).
        vocab_spec: Compact key (``content_hash:vocab_size``) of the
            vocabulary the JSD numbers were measured under; see
            `procgrep.bpe.VocabSpec`. They are only comparable to another
            report's when the keys match.
    """

    behavior_moved: bool
    fingerprint_jsd_to_target: tuple[float, float]
    outcome_delta: float
    verdict: Verdict
    vocab_spec: str | None = None


@dataclass(frozen=True)
class OptimizeReport:
    """What `optimize` changed and how much it helped on held-out traces.

    ``seed_val_score`` and ``best_val_score`` are the seed and tuned spec's
    discrimination (mean winner score minus mean loser score) on the held-out
    split. ``chosen`` summarizes the winning spec; ``train_trace`` is the
    objective after each accepted move; ``n_candidates`` is how many specs were
    scored.
    """

    seed_val_score: float
    best_val_score: float
    n_candidates: int
    chosen: dict[str, Any]
    train_trace: tuple[float, ...]


@overload
def enforce(
    spec: ProcedureSpec, mode: Literal["prompt"] = ..., scaffold: Scaffold | None = ...
) -> str: ...
@overload
def enforce(
    spec: ProcedureSpec, mode: Literal["guard"], scaffold: Scaffold | None = ...
) -> GuardArtifact: ...
@overload
def enforce(
    spec: ProcedureSpec, mode: Literal["decode"], scaffold: Scaffold | None = ...
) -> DecodeArtifact: ...
@overload
def enforce(
    spec: ProcedureSpec, mode: Literal["reward"], scaffold: Scaffold | None = ...
) -> RewardArtifact: ...
def enforce(
    spec: ProcedureSpec,
    mode: EnforceMode = "prompt",
    scaffold: Scaffold | None = None,
) -> str | GuardArtifact | RewardArtifact | DecodeArtifact:
    """Emit the rendered enforcement artifact for an external scaffold.

    This does not run an agent. procgrep is model-free; the caller wires
    the returned artifact into their own runner.

    * ``"prompt"`` returns system-prompt text. With ``scaffold=None`` it is
      the generic ruleset (`spec.to_prompt`). With ``scaffold="swe-agent"`` it
      is a SWE-agent config fragment (`scaffolds.to_swe_agent_config`); with
      ``scaffold="openhands"`` it is an OpenHands Skill markdown file
      (`scaffolds.to_openhands_skill`). The scaffold rendering wraps the same
      rule prose in that harness's native customization format.
    * ``"guard"`` returns a `GuardArtifact`: the guard patterns plus a
      streaming check callable that reports which guards a given atom
      prefix has already violated. ``scaffold`` does not change the artifact;
      see `scaffolds.to_swe_agent_config` for how the patterns map to a
      scaffold's control-flow / history-processing hook.
    * ``"reward"`` returns a `RewardArtifact`: a deterministic process reward
      (the full-trajectory `spec.score` plus per-step increments that sum to it)
      and the serialized scoring rules, for use as a dense RL signal. ``scaffold``
      is ignored.
    * ``"decode"`` returns a `DecodeArtifact`: the canonical atom alphabet plus
      an ``allowed(prefix)`` mask that forbids any next atom which would commit a
      penalty (an edit past the streak cap, or the final atom of a forbidden
      sequence). It constrains the action grammar, not text tokens, and never
      returns an empty set. Phases are reward/prompt signals, not decode masks.

    Raises:
        ValueError: For an unknown mode or an unknown scaffold.
    """
    if mode == "prompt":
        if scaffold is None:
            return spec.to_prompt()
        if scaffold == "swe-agent":
            return to_swe_agent_config(spec)
        if scaffold == "openhands":
            return to_openhands_skill(spec)
        raise ValueError(f"unknown scaffold {scaffold!r}; expected swe-agent, openhands, or None")
    if mode == "guard":
        patterns = tuple(spec.to_patterns())

        def check(prefix: Sequence[str]) -> list[str]:
            """Names of guards the atom prefix has already violated."""
            return _violated_guards(list(prefix), patterns)

        return GuardArtifact(patterns=patterns, check=check)
    if mode == "decode":
        alphabet = tuple(sorted(CANONICAL_ATOMS))
        edit_caps = [p.max_run for p in spec.penalties if p.max_run is not None]
        forbidden_seqs = [tuple(p.forbid_sequence) for p in spec.penalties if p.forbid_sequence]

        def allowed(prefix: Sequence[str]) -> set[str]:
            """Atoms that may follow ``prefix`` without committing a forbidden step."""
            seq = list(prefix)
            blocked: set[str] = set()
            if edit_caps:
                run = _trailing_run(seq, ATOM_EDIT)
                if any(run >= cap for cap in edit_caps):
                    blocked.add(ATOM_EDIT)
            for fseq in forbidden_seqs:
                if _ends_with(seq, fseq[:-1]):
                    blocked.add(fseq[-1])
            remaining = set(alphabet) - blocked
            return remaining or set(alphabet)

        return DecodeArtifact(
            alphabet=alphabet,
            allowed=allowed,
            rules_json=_decode_rules_json(alphabet, edit_caps, forbidden_seqs),
        )
    if mode == "reward":

        def reward(atoms: Sequence[str]) -> float:
            """Full-trajectory reward in ``[floor, ceiling]`` (= ``spec.score``)."""
            return spec.score(list(atoms)).score

        def step_rewards(atoms: Sequence[str]) -> list[float]:
            """Per-step reward increments; they sum to ``reward(atoms) - reward([])``.

            Each step is the change in clipped score as the prefix grows by one
            atom, giving a dense shaping signal rather than one terminal reward.
            """
            seq = list(atoms)
            prev = spec.score([]).score
            out: list[float] = []
            for i in range(len(seq)):
                cur = spec.score(seq[: i + 1]).score
                out.append(cur - prev)
                prev = cur
            return out

        return RewardArtifact(
            reward=reward, step_rewards=step_rewards, spec_json=_spec_scoring_json(spec)
        )
    raise ValueError(f"unknown enforce mode {mode!r}; expected prompt, guard, decode, or reward")


def verify(
    before: list[Trace],
    after: list[Trace],
    spec: ProcedureSpec,
    vocab: ProcedureVocabulary,
    *,
    outcome_field: str = "resolved",
    jsd_improvement_eps: float = 1e-3,
    outcome_improvement_eps: float = 0.0,
) -> VerifyReport:
    """Report the behavior x outcome 2x2 for an enforcement attempt.

    ``before`` and ``after`` are two trace populations a scaffold
    produced without and with the spec enforced. We measure two axes:

    * Behavior: the JSD from each population's mean procedure
      distribution to ``spec.target``. Behavior moved when the after-JSD
      is closer to the target than the before-JSD by more than
      ``jsd_improvement_eps``.
    * Outcome: the resolved rate (fraction of traces with a truthy
      ``outcome_field``) after minus before.

    The verdict crosses the two:

    * ``"lever"``: behavior moved toward the target and outcome improved.
    * ``"epiphenomenal"``: behavior moved but outcome did not. The spec
      changed how the agent acts without changing whether it succeeds.
    * ``"weak_enforcement"``: behavior did not move. Enforcement did not
      take, so the outcome axis is uninformative.

    Raises:
        ValueError: If ``spec.target`` is None (nothing to move toward).
    """
    if spec.target is None:
        raise ValueError(
            "verify needs spec.target to measure the behavior axis; derive the spec "
            "with ProcedureSpec.from_winners or set a target fingerprint"
        )

    target_dist = spec.target.distribution()
    before_jsd = _population_jsd_to(before, vocab, target_dist)
    after_jsd = _population_jsd_to(after, vocab, target_dist)

    behavior_moved = (before_jsd - after_jsd) > jsd_improvement_eps

    before_rate = _resolved_rate(before, outcome_field)
    after_rate = _resolved_rate(after, outcome_field)
    outcome_delta = after_rate - before_rate

    if not behavior_moved:
        verdict: Verdict = "weak_enforcement"
    elif outcome_delta > outcome_improvement_eps:
        verdict = "lever"
    else:
        verdict = "epiphenomenal"

    return VerifyReport(
        behavior_moved=behavior_moved,
        fingerprint_jsd_to_target=(round(before_jsd, 6), round(after_jsd, 6)),
        outcome_delta=round(outcome_delta, 6),
        verdict=verdict,
        vocab_spec=vocab.spec.compact(),
    )


def optimize(
    spec: ProcedureSpec,
    traces: list[Trace],
    *,
    outcome_field: str = "resolved",
    cap_grid: tuple[int, ...] = (3, 4, 5, 6, 8, 10),
    complexity_penalty: float = 0.02,
    val_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[ProcedureSpec, OptimizeReport]:
    """Tune a spec's penalty caps and phase set to better separate winners from losers.

    Offline and model-free: it scores existing labeled ``traces`` with each
    candidate spec; it never runs an agent. ``traces`` are split into
    winners/losers on ``outcome_field``; ``val_fraction`` is held out for
    reporting while coordinate ascent runs on the rest to maximize a
    discrimination objective — mean winner score minus mean loser score, less
    ``complexity_penalty`` per phase so the search does not simply accrue
    phases. Returns the tuned spec and an `OptimizeReport`.

    The search is two coordinated moves: pick the edit-streak cap from
    ``cap_grid`` that best separates the groups, then drop any phase whose
    removal improves the objective. Both are deterministic given ``seed``.

    This is the offline analog of a DSPy teleprompter: it proposes and scores
    candidate specs against observed traces. Running fresh agents under a
    candidate is the scaffold's job, not procgrep's; feed those traces back
    through `verify` to confirm a tuned spec is a lever, not an artifact.

    Raises:
        ValueError: when ``traces`` lack both a winner and a loser.
    """
    winners = [t for t in traces if bool(t.metadata.get(outcome_field))]
    losers = [t for t in traces if not bool(t.metadata.get(outcome_field))]
    if not winners or not losers:
        raise ValueError("optimize needs both winners and losers to discriminate against")

    train_w, val_w = _split(winners, val_fraction, seed)
    train_l, val_l = _split(losers, val_fraction, seed)

    def objective(candidate: ProcedureSpec) -> float:
        return _discrimination(candidate, train_w, train_l) - complexity_penalty * len(
            candidate.phases
        )

    best = spec
    trace: list[float] = []
    n_candidates = 0

    cap_idx = next((i for i, p in enumerate(best.penalties) if p.max_run is not None), None)
    if cap_idx is not None:
        ranked = []
        for cap in cap_grid:
            candidate = _with_cap(best, cap_idx, cap)
            ranked.append((objective(candidate), candidate))
            n_candidates += 1
        best = max(ranked, key=lambda pair: pair[0])[1]
        trace.append(objective(best))

    improved = True
    while improved and len(best.phases) > 1:
        improved = False
        base = objective(best)
        for i in range(len(best.phases)):
            candidate = _drop_phase(best, i)
            n_candidates += 1
            if objective(candidate) > base:
                best = candidate
                trace.append(objective(best))
                improved = True
                break

    chosen: dict[str, Any] = {
        "phases": [p.name for p in best.phases],
        "edit_streak_cap": next((p.max_run for p in best.penalties if p.max_run is not None), None),
    }
    report = OptimizeReport(
        seed_val_score=round(_discrimination(spec, val_w, val_l), 6),
        best_val_score=round(_discrimination(best, val_w, val_l), 6),
        n_candidates=n_candidates,
        chosen=chosen,
        train_trace=tuple(round(x, 6) for x in trace),
    )
    return best, report


def _discrimination(spec: ProcedureSpec, winners: list[Trace], losers: list[Trace]) -> float:
    """Mean winner score minus mean loser score under ``spec`` (0 if a side is empty)."""
    if not winners or not losers:
        return 0.0
    win = sum(spec.score(t).score for t in winners) / len(winners)
    lose = sum(spec.score(t).score for t in losers) / len(losers)
    return win - lose


def _split(items: list[Trace], val_fraction: float, seed: int) -> tuple[list[Trace], list[Trace]]:
    """Deterministic train/val split; neither side is empty when items exist."""
    import random

    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_val = int(len(items) * val_fraction) if len(items) > 1 else 0
    val = [items[i] for i in order[:n_val]]
    train = [items[i] for i in order[n_val:]]
    return (train or val), (val or train)


def _with_cap(spec: ProcedureSpec, penalty_idx: int, cap: int) -> ProcedureSpec:
    """Copy ``spec`` with one penalty's ``max_run`` set to ``cap``."""
    penalties = list(spec.penalties)
    penalties[penalty_idx] = replace(penalties[penalty_idx], max_run=cap)
    return replace(spec, penalties=tuple(penalties))


def _drop_phase(spec: ProcedureSpec, phase_idx: int) -> ProcedureSpec:
    """Copy ``spec`` without the phase at ``phase_idx``."""
    phases = tuple(p for i, p in enumerate(spec.phases) if i != phase_idx)
    return replace(spec, phases=phases)


def _trailing_run(atoms: list[str], atom: str) -> int:
    """Length of the contiguous run of ``atom`` at the end of ``atoms``."""
    n = 0
    for a in reversed(atoms):
        if a != atom:
            break
        n += 1
    return n


def _ends_with(atoms: list[str], suffix: tuple[str, ...]) -> bool:
    """True when ``atoms`` ends with ``suffix``; an empty suffix always matches."""
    if not suffix:
        return True
    return len(atoms) >= len(suffix) and tuple(atoms[-len(suffix) :]) == suffix


def _decode_rules_json(
    alphabet: tuple[str, ...],
    edit_caps: list[int],
    forbidden_seqs: list[tuple[str, ...]],
) -> str:
    """Serialize the decode constraint so an external sampler can apply it."""
    import json

    return json.dumps(
        {
            "alphabet": list(alphabet),
            "max_edit_run": min(edit_caps) if edit_caps else None,
            "forbidden_sequences": [list(s) for s in forbidden_seqs],
        },
        indent=2,
    )


def _spec_scoring_json(spec: ProcedureSpec) -> str:
    """Serialize the scoring-relevant rules of a spec to JSON.

    Emits phases, penalties, and the clip range — everything ``score`` needs —
    so a trainer can reload the reward without importing procgrep in its hot
    loop. The target fingerprint is omitted: it drives enforcement and verify,
    not scoring.
    """
    import json

    return json.dumps(
        {
            "name": spec.name,
            "floor": spec.floor,
            "ceiling": spec.ceiling,
            "phases": [
                {
                    "name": p.name,
                    "reward": p.reward,
                    "require_any": list(p.require_any),
                    "before_first": p.before_first,
                    "min_count": p.min_count,
                }
                for p in spec.phases
            ],
            "penalties": [
                {
                    "name": p.name,
                    "reward": p.reward,
                    "max_run": p.max_run,
                    "forbid_sequence": list(p.forbid_sequence),
                }
                for p in spec.penalties
            ],
        },
        indent=2,
    )


def _violated_guards(prefix: list[str], patterns: tuple[Pattern, ...]) -> list[str]:
    """Names of must-hold-false guards whose regex matches the prefix.

    Reuses the same encoding `patterns.match_patterns` uses: the
    space-joined atom string with a trailing space.
    """
    import re

    encoded = " ".join(prefix) + " "
    return [p.name for p in patterns if re.search(p.pattern, encoded) is not None]


def _population_jsd_to(
    traces: list[Trace],
    vocab: ProcedureVocabulary,
    target_dist: Any,
) -> float:
    """JSD from a population's count-summed mean distribution to a target.

    An empty population is maximally far (JSD 1.0 under base-2), so a
    spec that produces no after-traces never reads as having moved.
    """
    if not traces:
        return 1.0
    fps = encode(traces, vocab=vocab)
    totals = np.zeros(vocab.size, dtype=np.float64)
    for fp in fps:
        totals += np.asarray(fp.counts, dtype=np.float64)
    return jsd(totals, target_dist)


def _resolved_rate(traces: list[Trace], outcome_field: str) -> float:
    """Fraction of traces with a truthy ``outcome_field`` in metadata.

    Traces missing the field count as unresolved. An empty population is
    rate 0.0.
    """
    if not traces:
        return 0.0
    resolved = sum(1 for t in traces if bool(t.metadata.get(outcome_field)))
    return resolved / len(traces)


__all__ = [
    "DecodeArtifact",
    "EnforceMode",
    "GuardArtifact",
    "OptimizeReport",
    "RewardArtifact",
    "Scaffold",
    "Verdict",
    "VerifyReport",
    "enforce",
    "optimize",
    "verify",
]
