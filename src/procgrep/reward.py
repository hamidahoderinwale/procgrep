"""Declarative procedural specs: the reward role of the programmability loop.

A `ProcedureSpec` is the unit of the derive -> specify -> enforce -> score ->
verify loop. It declares the procedural shape a trajectory should have: ordered
phases (milestones that earn reward), penalties (failure-mode patterns that
deduct), a clipping range, and an optional target fingerprint that enforcement
aims a population at.

Design decisions:

* Typed dataclasses, not a free-form dict. Benefit: a misspelled key fails at
  load time (`from_yaml`) instead of silently scoring zero, which was the
  "stringly-typed island" the old dict spec created. Price: a fixed schema, so
  exotic one-off rules need a code change rather than a YAML field.
* `score` returns both the clipped score and the raw unclipped total. Benefit:
  a caller training an RL signal can see how far past the ceiling (or below the
  floor) a trajectory went, which clipping would hide. Price: one more field on
  the result.
* The spec is the single source of truth for three renderings: a reward
  (`score`), a prompt (`to_prompt`), and guard patterns (`to_patterns`). Benefit:
  derive once, enforce many ways. Price: each renderer is a partial view (a
  prompt cannot express a contiguous-run penalty as precisely as a regex).
* `from_winners` derives a spec from passing trajectories rather than asking a
  human to author one. Benefit: the spec is grounded in observed behavior.
  Price: it is only as good as the corpus, and it captures correlation with
  passing, not causation.

A thin back-compat shim keeps the old `score(atoms, dict)` and `load_spec`
signatures working for case-study callers that pass a parsed dict.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from procgrep.bpe import ProcedureVocabulary
from procgrep.encode import Fingerprint, encode
from procgrep.patterns import Pattern
from procgrep.stats import discriminative_procedures
from procgrep.types import (
    ATOM_EDIT,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    PROCEDURE_SEPARATOR,
    Trace,
)

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass(frozen=True)
class Phase:
    """One procedural milestone that earns reward when satisfied.

    A phase is satisfied when its required atoms appear (at least
    ``min_count`` times), optionally restricted to the prefix before the
    first occurrence of ``before_first``. ``require_any`` is an OR over
    atoms: any one of them appearing enough times satisfies the phase.
    """

    name: str
    reward: float
    require_any: tuple[str, ...] = ()
    before_first: str | None = None
    min_count: int = 1


@dataclass(frozen=True)
class Penalty:
    """One failure-mode pattern that deducts reward when it fires.

    Two independent triggers, either of which fires the penalty:
    ``max_run`` deducts when an atom repeats contiguously more than that
    many times; ``forbid_sequence`` deducts when that exact contiguous
    atom sequence appears.
    """

    name: str
    reward: float
    max_run: int | None = None
    forbid_sequence: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewardResult:
    """Outcome of scoring a trajectory against a `ProcedureSpec`.

    ``score`` is clipped to ``[floor, ceiling]``; ``raw`` is the
    unclipped sum of phase rewards minus penalties. Compare the two to
    see whether clipping was load-bearing.
    """

    score: float
    raw: float
    phase_scores: dict[str, float]
    penalties: dict[str, float]
    satisfied_phases: list[str]
    triggered_penalties: list[str]


@dataclass(frozen=True)
class ProcedureSpec:
    """A declarative procedural spec: phases, penalties, range, target.

    The unit of the programmability loop. Derive it from winners
    (`from_winners`), load it from YAML (`from_yaml`), then render it as a
    reward (`score`), a system prompt (`to_prompt`), or guard patterns
    (`to_patterns`).
    """

    phases: tuple[Phase, ...] = ()
    penalties: tuple[Penalty, ...] = ()
    floor: float = 0.0
    ceiling: float = 1.0
    target: Fingerprint | None = None
    name: str = "procedure_spec"

    def score(self, trace: Trace | Sequence[str]) -> RewardResult:
        """Score a trajectory in ``[floor, ceiling]`` with a raw total too.

        Accepts a `Trace` or a bare atom sequence. Phases add their
        reward when satisfied; penalties subtract theirs when they fire.
        """
        atoms = list(trace.atoms) if isinstance(trace, Trace) else list(trace)

        phase_scores: dict[str, float] = {}
        satisfied: list[str] = []
        penalty_scores: dict[str, float] = {}
        triggered: list[str] = []
        raw = 0.0

        for phase in self.phases:
            if _phase_satisfied(atoms, phase):
                phase_scores[phase.name] = phase.reward
                satisfied.append(phase.name)
                raw += phase.reward
            else:
                phase_scores[phase.name] = 0.0

        for penalty in self.penalties:
            if _penalty_fires(atoms, penalty):
                penalty_scores[penalty.name] = -penalty.reward
                triggered.append(penalty.name)
                raw -= penalty.reward

        clipped = max(self.floor, min(self.ceiling, raw))
        return RewardResult(
            score=round(clipped, 4),
            raw=round(raw, 4),
            phase_scores=phase_scores,
            penalties=penalty_scores,
            satisfied_phases=satisfied,
            triggered_penalties=triggered,
        )

    def to_prompt(self) -> str:
        """Render the spec as a natural-language system-prompt ruleset.

        A scaffold can inject the returned text as a system prompt. This
        is a lossy view: it states the procedural intent in prose, not
        the exact scoring arithmetic.
        """
        lines: list[str] = [
            "You are a coding agent. Follow this procedure when solving a task.",
            "",
        ]
        if self.phases:
            lines.append("Do, in order:")
            for i, phase in enumerate(self.phases, start=1):
                lines.append(f"  {i}. {_phase_sentence(phase)}")
            lines.append("")
        if self.penalties:
            lines.append("Avoid:")
            for penalty in self.penalties:
                lines.append(f"  - {_penalty_sentence(penalty)}")
            lines.append("")
        lines.append(
            "These rules reward structured exploration, test-driven editing, "
            "and verification before submitting."
        )
        return "\n".join(lines)

    def to_patterns(self) -> list[Pattern]:
        """Render penalties as guard `Pattern`s reusing `patterns.py`.

        Each penalty becomes a ``must_hold=False`` regex over the
        space-joined atom sequence: a match means the trajectory
        violated the guard. Phases are not rendered (a phase is a
        positive reward signal, not a hard guard).
        """
        out: list[Pattern] = []
        for penalty in self.penalties:
            regex = _penalty_regex(penalty)
            if regex is None:
                continue
            out.append(
                Pattern(
                    name=penalty.name,
                    description=_penalty_sentence(penalty),
                    pattern=regex,
                    must_hold=False,
                )
            )
        return out

    @classmethod
    def from_winners(
        cls,
        traces: Iterable[Trace],
        vocab: ProcedureVocabulary,
        *,
        k: int = 10,
        outcome_field: str = "resolved",
    ) -> ProcedureSpec:
        """Derive a spec from passing trajectories.

        Splits ``traces`` into winners and losers on the boolean
        ``outcome_field`` in each trace's metadata, then builds the spec
        from two sources:

        * The procedures that separate winners from losers
          (`discriminative_procedures`, ranked by log-odds). Procedures
          that favor winners and consist of meaningful atoms become
          phases.
        * Simple action-level stats over the winners: the edit-streak cap
          is set to the longest contiguous edit run seen in any winner
          (so a penalty fires only beyond the passing distribution), and
          a test-after-edit phase is required when winners reliably test
          after editing.

        The target fingerprint is the population we want enforcement to
        move toward: the (count-summed) mean of the winners under
        ``vocab``.

        Design note: this derives correlation with passing, not a causal
        recipe. Validate the derived spec before treating it as a lever
        (see `procgrep.program.verify`).
        """
        all_traces = list(traces)
        winners = [t for t in all_traces if bool(t.metadata.get(outcome_field))]
        losers = [t for t in all_traces if not bool(t.metadata.get(outcome_field))]
        if not winners:
            raise ValueError(
                f"no winners found: no trace has a truthy {outcome_field!r} in metadata"
            )

        phases: list[Phase] = []

        winner_disc = _winner_procedures(winners, losers, vocab, k=k)
        for rank, proc in enumerate(winner_disc):
            phases.append(
                Phase(
                    name=f"procedure_{rank + 1}_{_safe(proc)}",
                    reward=round(0.5 / max(len(winner_disc), 1), 4),
                    require_any=(proc,),
                    min_count=1,
                )
            )

        if _winners_test_after_edit(winners):
            phases.append(
                Phase(
                    name="verification",
                    reward=0.25,
                    require_any=(ATOM_RUN_TEST,),
                    min_count=1,
                )
            )

        if _winners_explore_before_edit(winners):
            phases.append(
                Phase(
                    name="exploration",
                    reward=0.1,
                    require_any=(ATOM_SEARCH_REPO, ATOM_READ_FILE),
                    before_first=ATOM_EDIT,
                    min_count=1,
                )
            )

        penalties: list[Penalty] = []
        cap = _winner_edit_streak_cap(winners)
        if cap is not None:
            penalties.append(Penalty(name="edit_streak", reward=0.15, max_run=cap))

        target = _winner_target(winners, vocab)
        return cls(
            phases=tuple(phases),
            penalties=tuple(penalties),
            floor=0.0,
            ceiling=1.0,
            target=target,
            name="derived_from_winners",
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProcedureSpec:
        """Load and validate a spec from YAML; a typo raises here.

        Every key is validated against the dataclass schema at load
        time. An unrecognized key (a typo such as ``min_counts``) raises
        `ValueError` rather than being silently dropped and scoring zero.
        """
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is required: pip install pyyaml")
        raw = _yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: Any, *, source: str = "<dict>") -> ProcedureSpec:
        """Build and validate a spec from an already-parsed mapping."""
        if not isinstance(raw, dict):
            raise TypeError(f"spec in {source} must be a mapping, got {type(raw).__name__}")

        _reject_unknown_keys(raw, _SPEC_KEYS, where=f"spec in {source}")

        phases = tuple(
            _phase_from_dict(entry, source=source) for entry in raw.get("phases", []) or []
        )
        penalties = tuple(
            _penalty_from_dict(entry, source=source) for entry in raw.get("penalties", []) or []
        )
        return cls(
            phases=phases,
            penalties=penalties,
            floor=float(raw.get("floor", 0.0)),
            ceiling=float(raw.get("ceiling", 1.0)),
            target=None,
            name=str(raw.get("name", "procedure_spec")),
        )


_SPEC_KEYS = frozenset({"name", "description", "phases", "penalties", "floor", "ceiling"})
_PHASE_KEYS = frozenset({"name", "reward", "require_any", "before_first", "min_count"})
_PENALTY_KEYS = frozenset({"name", "reward", "max_run", "forbid_sequence"})


def _reject_unknown_keys(entry: dict[str, Any], allowed: frozenset[str], *, where: str) -> None:
    """Raise if ``entry`` carries a key outside ``allowed``.

    This is the load-time typo guard. A field name that does not match
    the schema is almost always a mistake the author wants to hear about
    immediately, not a silently ignored no-op.
    """
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(
            f"{where} has unrecognized keys {sorted(unknown)}; allowed keys are {sorted(allowed)}"
        )


def _require(entry: dict[str, Any], key: str, *, where: str) -> Any:
    if key not in entry:
        raise ValueError(f"{where} is missing required key {key!r}")
    return entry[key]


def _phase_from_dict(entry: Any, *, source: str) -> Phase:
    if not isinstance(entry, dict):
        raise TypeError(f"phase in {source} is not a mapping: {entry!r}")
    where = f"phase {entry.get('name', '?')!r} in {source}"
    _reject_unknown_keys(entry, _PHASE_KEYS, where=where)
    return Phase(
        name=str(_require(entry, "name", where=where)),
        reward=float(_require(entry, "reward", where=where)),
        require_any=tuple(str(a) for a in entry.get("require_any", ()) or ()),
        before_first=(str(entry["before_first"]) if entry.get("before_first") else None),
        min_count=int(entry.get("min_count", 1)),
    )


def _penalty_from_dict(entry: Any, *, source: str) -> Penalty:
    if not isinstance(entry, dict):
        raise TypeError(f"penalty in {source} is not a mapping: {entry!r}")
    where = f"penalty {entry.get('name', '?')!r} in {source}"
    _reject_unknown_keys(entry, _PENALTY_KEYS, where=where)
    max_run = entry.get("max_run")
    return Penalty(
        name=str(_require(entry, "name", where=where)),
        reward=float(_require(entry, "reward", where=where)),
        max_run=(int(max_run) if max_run is not None else None),
        forbid_sequence=tuple(str(a) for a in entry.get("forbid_sequence", ()) or ()),
    )


def _first_index(atoms: list[str], atom: str) -> int:
    try:
        return atoms.index(atom)
    except ValueError:
        return len(atoms)


def _phase_satisfied(atoms: list[str], phase: Phase) -> bool:
    """True when at least one required atom clears ``min_count`` in scope."""
    if not phase.require_any:
        return True
    scope = atoms[: _first_index(atoms, phase.before_first)] if phase.before_first else atoms
    return any(scope.count(a) >= phase.min_count for a in phase.require_any)


def _max_contiguous_run(atoms: list[str], atom: str) -> int:
    """Longest contiguous run of ``atom`` in ``atoms``."""
    best = 0
    run = 0
    for a in atoms:
        if a == atom:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _has_contiguous(atoms: list[str], seq: tuple[str, ...]) -> bool:
    n, p = len(atoms), len(seq)
    if p == 0:
        return False
    return any(tuple(atoms[i : i + p]) == seq for i in range(n - p + 1))


def _penalty_fires(atoms: list[str], penalty: Penalty) -> bool:
    """True when either of the penalty's triggers matches."""
    if penalty.max_run is not None and _max_contiguous_run(atoms, ATOM_EDIT) > penalty.max_run:
        return True
    return bool(penalty.forbid_sequence) and _has_contiguous(atoms, penalty.forbid_sequence)


def _phase_sentence(phase: Phase) -> str:
    atoms = " or ".join(_humanize(a) for a in phase.require_any) or "act"
    when = f" before your first {_humanize(phase.before_first)}" if phase.before_first else ""
    count = f" at least {phase.min_count} times" if phase.min_count > 1 else ""
    return f"{atoms}{count}{when}."


def _penalty_sentence(penalty: Penalty) -> str:
    parts: list[str] = []
    if penalty.max_run is not None:
        parts.append(f"more than {penalty.max_run} edits in a row without running tests")
    if penalty.forbid_sequence:
        parts.append("the sequence " + " then ".join(_humanize(a) for a in penalty.forbid_sequence))
    return "; ".join(parts) if parts else penalty.name


def _penalty_regex(penalty: Penalty) -> str | None:
    """Regex over the space-joined atom string, or None if unrenderable."""
    if penalty.forbid_sequence:
        body = " ".join(re_escape(a) for a in penalty.forbid_sequence)
        return f"(?:^| ){body}(?: |$)"
    if penalty.max_run is not None:
        # A run longer than max_run edits: max_run + 1 contiguous edits.
        return f"(?:{re_escape(ATOM_EDIT)} ){{{penalty.max_run + 1},}}"
    return None


def re_escape(atom: str) -> str:
    """Escape a single atom for safe embedding in a regex."""
    return re.escape(atom)


def _humanize(atom: str | None) -> str:
    if atom is None:
        return ""
    return atom.replace(PROCEDURE_SEPARATOR, " then ").replace("_", " ")


def _safe(token: str) -> str:
    """A token slug safe for a phase name."""
    return token.replace(PROCEDURE_SEPARATOR, "_").replace(" ", "_")


def _winner_procedures(
    winners: list[Trace],
    losers: list[Trace],
    vocab: ProcedureVocabulary,
    *,
    k: int,
) -> list[str]:
    """Procedures that favor winners over losers, most discriminative first.

    Returns multi-atom procedures (or meaningful single atoms) whose
    log-odds favor the winners. Falls back to an empty list when there
    are no losers to contrast against.
    """
    if not losers:
        return []
    tagged = _tag_for_contrast(winners, losers)
    fps = encode(tagged, vocab=vocab)
    disc = discriminative_procedures(
        fps,
        vocab,
        group_a="win",
        group_b="lose",
        k=max(k * 2, k),
        ranking="log_odds",
        group_by="group",
    )
    out: list[str] = []
    for row in disc:
        if row.log_odds <= 0.0:
            continue
        out.append(row.procedure)
        if len(out) >= k:
            break
    return out


def _tag_for_contrast(winners: list[Trace], losers: list[Trace]) -> list[Trace]:
    """Copy traces with group set to ``"win"`` / ``"lose"`` for contrast."""
    tagged: list[Trace] = []
    for t in winners:
        tagged.append(
            Trace(
                trace_id=t.trace_id, agent=t.agent, atoms=t.atoms, group="win", metadata=t.metadata
            )
        )
    for t in losers:
        tagged.append(
            Trace(
                trace_id=t.trace_id, agent=t.agent, atoms=t.atoms, group="lose", metadata=t.metadata
            )
        )
    return tagged


def _winner_edit_streak_cap(winners: list[Trace]) -> int | None:
    """Longest contiguous edit run seen across winners, or None.

    The penalty cap is set here so a streak penalty only fires beyond
    the passing distribution.
    """
    runs = [_max_contiguous_run(list(t.atoms), ATOM_EDIT) for t in winners]
    runs = [r for r in runs if r > 0]
    if not runs:
        return None
    return max(runs)


def _winners_test_after_edit(winners: list[Trace], *, fraction: float = 0.5) -> bool:
    """True when most winners run a test at some point after an edit."""
    hits = 0
    eligible = 0
    for t in winners:
        atoms = list(t.atoms)
        if ATOM_EDIT not in atoms:
            continue
        eligible += 1
        first_edit = atoms.index(ATOM_EDIT)
        if ATOM_RUN_TEST in atoms[first_edit + 1 :]:
            hits += 1
    if eligible == 0:
        return False
    return hits / eligible >= fraction


def _winners_explore_before_edit(winners: list[Trace], *, fraction: float = 0.5) -> bool:
    """True when most winners search or read before their first edit."""
    hits = 0
    eligible = 0
    for t in winners:
        atoms = list(t.atoms)
        if ATOM_EDIT not in atoms:
            continue
        eligible += 1
        before = atoms[: atoms.index(ATOM_EDIT)]
        if ATOM_SEARCH_REPO in before or ATOM_READ_FILE in before:
            hits += 1
    if eligible == 0:
        return False
    return hits / eligible >= fraction


def _winner_target(winners: list[Trace], vocab: ProcedureVocabulary) -> Fingerprint:
    """Count-summed mean of the winners as a single target fingerprint.

    The target is the procedure distribution enforcement aims at. We sum
    counts across winners so the result is itself a valid `Fingerprint`
    (integer counts) carrying the population's procedural shape.
    """
    fps = encode(winners, vocab=vocab)
    size = vocab.size
    totals = [0] * size
    for fp in fps:
        for i, c in enumerate(fp.counts):
            totals[i] += c
    return Fingerprint(
        trace_id="__winner_target__",
        agent="__target__",
        group="__target__",
        counts=tuple(totals),
    )


# Back-compat shim for the old dict-based API.
#
# Earlier callers used `load_spec(path) -> dict` plus `score(atoms, dict)`.
# The case studies still pass a parsed dict. We keep those signatures working by
# detecting a dict second argument and routing through `ProcedureSpec`. New code
# should use `ProcedureSpec.from_yaml` and `spec.score`.


def load_spec(path: str | Path) -> dict[str, Any]:
    """Deprecated. Load a raw spec dict; prefer `ProcedureSpec.from_yaml`.

    Kept so existing case-study callers that consume a parsed dict keep
    working. Validation happens when the dict is handed to `score`.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")
    result: dict[str, Any] = _yaml.safe_load(Path(path).read_text())
    return result


def score(atoms: Sequence[str], spec: ProcedureSpec | dict[str, Any]) -> RewardResult:
    """Score atoms against a `ProcedureSpec` or a legacy spec dict.

    The legacy path adapts the old dict schema (phases with
    ``require_any``/``require_pattern``/``require_sequence``, penalties,
    bonuses) into a `ProcedureSpec` so historical callers keep working.
    Prefer `ProcedureSpec.score`.
    """
    if isinstance(spec, ProcedureSpec):
        return spec.score(list(atoms))
    return _score_legacy_dict(list(atoms), spec)


def _score_legacy_dict(atoms: list[str], spec: dict[str, Any]) -> RewardResult:
    """Adapt and score the legacy dict spec.

    Supports the subset of the old schema the example file and case
    studies use: phase ``require_any`` with ``min_occurrences`` and
    ``before_first``, contiguous penalties, and bonuses (treated as
    extra phases). Phase types this typed schema cannot express
    (``require_sequence``, ``require_pattern``) are evaluated inline so
    the example spec still scores as before.
    """
    phase_scores: dict[str, float] = {}
    satisfied: list[str] = []
    penalty_scores: dict[str, float] = {}
    triggered: list[str] = []
    raw = 0.0

    for phase in spec.get("phases", []) or []:
        name = phase["name"]
        reward = float(phase.get("reward", 0.0))
        if _legacy_phase_ok(atoms, phase):
            phase_scores[name] = reward
            satisfied.append(name)
            raw += reward
        else:
            phase_scores[name] = 0.0

    for penalty in spec.get("penalties", []) or []:
        name = penalty["name"]
        amount = float(penalty.get("penalty", 0.0))
        if _legacy_penalty_fires(atoms, penalty):
            penalty_scores[name] = -amount
            triggered.append(name)
            raw -= amount

    for bonus in spec.get("bonuses", []) or []:
        name = bonus["name"]
        reward = float(bonus.get("reward", 0.0))
        if _legacy_bonus_ok(atoms, bonus):
            phase_scores[name] = reward
            satisfied.append(name)
            raw += reward

    floor = float(spec.get("floor", 0.0))
    ceiling = float(spec.get("ceiling", 1.0))
    clipped = max(floor, min(ceiling, raw))
    return RewardResult(
        score=round(clipped, 4),
        raw=round(raw, 4),
        phase_scores=phase_scores,
        penalties=penalty_scores,
        satisfied_phases=satisfied,
        triggered_penalties=triggered,
    )


def _legacy_scope(atoms: list[str], entry: dict[str, Any]) -> list[str]:
    bf = entry.get("before_first")
    return atoms[: _first_index(atoms, bf)] if bf else atoms


def _legacy_subsequence(atoms: list[str], seq: list[str], max_gap: int) -> bool:
    if not seq:
        return True
    if len(seq) == 1:
        return seq[0] in atoms
    head, rest = seq[0], seq[1:]
    for i, a in enumerate(atoms):
        if a == head and _legacy_subsequence(atoms[i + 1 : i + 1 + max_gap], rest, max_gap):
            return True
    return False


def _legacy_phase_ok(atoms: list[str], phase: dict[str, Any]) -> bool:
    scope = _legacy_scope(atoms, phase)
    if "require_any" in phase:
        min_occ = int(phase.get("min_occurrences", 1))
        names = [r["atom"] for r in phase["require_any"] if "atom" in r]
        if not any(scope.count(a) >= min_occ for a in names):
            return False
    if "require_pattern" in phase and not any(a in scope for a in phase["require_pattern"]):
        return False
    if "require_sequence" in phase:
        max_gap = int(phase.get("max_gap", 999))
        min_occ = int(phase.get("min_occurrences", 1))
        found = sum(
            1
            for i in range(len(atoms))
            if _legacy_subsequence(atoms[i:], phase["require_sequence"], max_gap)
        )
        if found < min_occ:
            return False
    return True


def _legacy_penalty_fires(atoms: list[str], penalty: dict[str, Any]) -> bool:
    if "pattern" in penalty and penalty.get("contiguous", False):
        return _has_contiguous(atoms, tuple(penalty["pattern"]))
    if "require_absent_before" in penalty:
        before = atoms[: _first_index(atoms, penalty.get("before_first", "edit"))]
        return not any(a in before for a in penalty["require_absent_before"])
    return False


def _legacy_bonus_ok(atoms: list[str], bonus: dict[str, Any]) -> bool:
    if "require_sequence" in bonus:
        scope = _legacy_scope(atoms, bonus)
        return _legacy_subsequence(scope, bonus["require_sequence"], int(bonus.get("max_gap", 999)))
    return False


__all__ = [
    "Penalty",
    "Phase",
    "ProcedureSpec",
    "RewardResult",
    "load_spec",
    "score",
]
