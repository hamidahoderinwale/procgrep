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
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from procgrep.bpe import ProcedureVocabulary
from procgrep.encode import encode
from procgrep.jsd import jsd
from procgrep.patterns import Pattern
from procgrep.reward import ProcedureSpec
from procgrep.types import Trace

EnforceMode = Literal["prompt", "guard", "decode", "reward"]
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
    """

    behavior_moved: bool
    fingerprint_jsd_to_target: tuple[float, float]
    outcome_delta: float
    verdict: Verdict


def enforce(
    spec: ProcedureSpec,
    mode: EnforceMode = "prompt",
) -> str | GuardArtifact:
    """Emit the rendered enforcement artifact for an external scaffold.

    This does not run an agent. procgrep is model-free; the caller wires
    the returned artifact into their own runner.

    * ``"prompt"`` returns the system-prompt text (`spec.to_prompt`).
    * ``"guard"`` returns a `GuardArtifact`: the guard patterns plus a
      streaming check callable that reports which guards a given atom
      prefix has already violated.
    * ``"decode"`` and ``"reward"`` are not implemented yet.

    Raises:
        NotImplementedError: For ``"decode"`` and ``"reward"``.
        ValueError: For an unknown mode.
    """
    if mode == "prompt":
        return spec.to_prompt()
    if mode == "guard":
        patterns = tuple(spec.to_patterns())

        def check(prefix: Sequence[str]) -> list[str]:
            """Names of guards the atom prefix has already violated."""
            return _violated_guards(list(prefix), patterns)

        return GuardArtifact(patterns=patterns, check=check)
    if mode == "decode":
        raise NotImplementedError(
            "decode-time enforcement is on the roadmap: compile the spec to a "
            "constrained-decoding mask the scaffold applies at the token boundary."
        )
    if mode == "reward":
        raise NotImplementedError(
            "reward-shaping enforcement is on the roadmap: hand spec.score to the "
            "trainer as a dense per-step signal rather than a system prompt."
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
    )


def optimize(
    spec: ProcedureSpec,
    traces: list[Trace],
    vocab: ProcedureVocabulary,
    *,
    metric: Callable[[VerifyReport], float] | None = None,
    n_candidates: int = 16,
) -> ProcedureSpec:
    """Search the spec space for a stronger lever. Not implemented yet.

    Roadmap: the DSPy-teleprompter analog for procedural specs. Propose
    candidate specs (perturb phase rewards, penalty caps, the phase set),
    enforce each as a prompt, collect the resulting traces from the
    scaffold, score every candidate with `verify`, and keep the spec that
    maximizes ``metric`` over the verify report. The model-free
    constraint holds: optimize proposes and scores specs; the scaffold,
    not procgrep, runs the agent that produces the candidate traces.
    """
    raise NotImplementedError(
        "optimize is on the roadmap: a DSPy-teleprompter analog that proposes "
        "candidate ProcedureSpecs, has the external scaffold run each, and keeps "
        "the spec whose verify report scores highest. procgrep stays model-free; "
        "it proposes and scores specs, the scaffold runs the agent."
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
    "EnforceMode",
    "GuardArtifact",
    "Verdict",
    "VerifyReport",
    "enforce",
    "optimize",
    "verify",
]
