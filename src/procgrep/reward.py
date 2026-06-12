"""Procedural reward scoring from YAML trajectory specifications.

A reward spec defines phases (procedural milestones), penalties (known
failure-mode patterns), and bonuses (best-practice signals). Scoring a
trajectory against a spec returns a [0, 1] partial reward that supplements
binary pass/fail with procedural signal useful for RL training.

Usage::

    from procgrep.reward import load_spec, score, RewardResult

    spec = load_spec("examples/rules/reward_spec_swe_agent.yaml")
    atoms = ["search_repo", "read_file", "think", "edit", "run_test", "submit"]
    result = score(atoms, spec)
    print(result.proc_score)       # 0.70
    print(result.satisfied_phases) # ["exploration", "diagnosis", ...]

Spec format (YAML)::

    name: my_spec
    phases:
      - name: exploration
        reward: 0.10
        require_any:
          - atom: search_repo
          - atom: read_file
        min_occurrences: 2
        before_first: edit
    penalties:
      - name: edit_streak
        pattern: [edit, edit, edit, edit, edit]
        contiguous: true
        penalty: 0.15
    bonuses:
      - name: test_driven
        require_sequence: [run_test]
        before_first: edit
        reward: 0.10
    floor: 0.0
    ceiling: 1.0

See ``examples/rules/reward_spec_swe_agent.yaml`` for a full worked example.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# Data classes.


@dataclass(frozen=True)
class RewardResult:
    """Outcome of scoring a trajectory against a reward spec.

    Attributes:
        proc_score: Clipped [floor, ceiling] total procedural score.
        phase_scores: Per-phase reward earned (0.0 if not satisfied).
        penalties: Per-penalty deduction applied (negative values).
        bonuses: Per-bonus reward earned.
        satisfied_phases: Names of phases whose conditions were met.
        triggered_penalties: Names of penalties that fired.
        triggered_bonuses: Names of bonuses that were earned.
    """

    proc_score: float
    phase_scores: dict[str, float]
    penalties: dict[str, float]
    bonuses: dict[str, float]
    satisfied_phases: list[str]
    triggered_penalties: list[str]
    triggered_bonuses: list[str]


# Spec loading.


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load a reward spec from a YAML file.

    Args:
        path: Path to the YAML reward spec file.

    Returns:
        Parsed spec dict ready for use with :func:`score`.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If the spec file does not exist.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required: pip install pyyaml")
    result: dict[str, Any] = _yaml.safe_load(Path(path).read_text())
    return result


# Pattern matching.


def _first_occurrence(atoms: list[str], atom: str) -> int:
    try:
        return atoms.index(atom)
    except ValueError:
        return len(atoms)


def _atoms_before_first(atoms: list[str], target: str) -> list[str]:
    return atoms[: _first_occurrence(atoms, target)]


def _has_any_atom(atoms: list[str], atom_list: list[str], min_occurrences: int = 1) -> bool:
    return any(atoms.count(a) >= min_occurrences for a in atom_list)


def _has_contiguous_pattern(atoms: list[str], pattern: list[str]) -> bool:
    n, p = len(atoms), len(pattern)
    return any(atoms[i : i + p] == pattern for i in range(n - p + 1))


def _has_sequence_within_gap(atoms: list[str], seq: list[str], max_gap: int = 999) -> bool:
    """True if seq appears as an ordered subsequence within max_gap steps."""
    if len(seq) < 2:
        return seq[0] in atoms if seq else True
    a = seq[0]
    rest = seq[1:]
    for i, atom in enumerate(atoms):
        if atom == a:
            window = atoms[i + 1 : i + 1 + max_gap]
            if _has_sequence_within_gap(window, rest, max_gap):
                return True
    return False


# Phase / penalty / bonus evaluation.


def _eval_phase(atoms: list[str], phase: dict[str, Any]) -> bool:
    before_first = phase.get("before_first")
    scope = _atoms_before_first(atoms, before_first) if before_first else atoms

    if "require_any" in phase:
        min_occ = phase.get("min_occurrences", 1)
        atom_names = [r["atom"] for r in phase["require_any"] if "atom" in r]
        if not _has_any_atom(scope, atom_names, min_occ):
            return False

    if "require_pattern" in phase and not any(a in scope for a in phase["require_pattern"]):
        return False

    if "require_sequence" in phase:
        seq = phase["require_sequence"]
        max_gap = phase.get("max_gap", 999)
        min_occ = phase.get("min_occurrences", 1)
        found = sum(
            1 for i in range(len(atoms)) if _has_sequence_within_gap(atoms[i:], seq, max_gap)
        )
        if found < min_occ:
            return False

    if "require_absent_before" in phase:
        required_absent = phase["require_absent_before"]
        before = _atoms_before_first(atoms, phase.get("before_first", "edit"))
        if any(a in before for a in required_absent):
            return False

    return True


def _eval_penalty(atoms: list[str], penalty: dict[str, Any]) -> bool:
    if "pattern" in penalty and penalty.get("contiguous", False):
        return _has_contiguous_pattern(atoms, penalty["pattern"])
    if "require_absent_before" in penalty:
        required_absent = penalty["require_absent_before"]
        target = penalty.get("before_first", "edit")
        before = _atoms_before_first(atoms, target)
        return not any(a in before for a in required_absent)
    return False


def _eval_bonus(atoms: list[str], bonus: dict[str, Any]) -> bool:
    if "require_sequence" in bonus:
        seq = bonus["require_sequence"]
        before_first = bonus.get("before_first")
        scope = _atoms_before_first(atoms, before_first) if before_first else atoms
        max_gap = bonus.get("max_gap", 999)
        return _has_sequence_within_gap(scope, seq, max_gap)
    return False


# Main scoring function.


def score(atoms: list[str], spec: dict[str, Any]) -> RewardResult:
    """Score a canonical atom sequence against a reward spec.

    Args:
        atoms: Canonical atom sequence for one trajectory
               (e.g. ``["search_repo", "read_file", "edit", "run_test"]``).
        spec: Parsed reward spec dict (from :func:`load_spec` or inline).

    Returns:
        :class:`RewardResult` with the total procedural score and breakdown.

    Example::

        spec = load_spec("examples/rules/reward_spec_swe_agent.yaml")
        result = score(["search_repo", "read_file", "edit", "run_test", "submit"], spec)
        assert result.proc_score > 0
    """
    phase_scores: dict[str, float] = {}
    penalty_scores: dict[str, float] = {}
    bonus_scores: dict[str, float] = {}
    satisfied: list[str] = []
    triggered_penalties: list[str] = []
    triggered_bonuses: list[str] = []
    total = 0.0

    for phase in spec.get("phases", []):
        name = phase["name"]
        reward = float(phase.get("reward", 0.0))
        if _eval_phase(atoms, phase):
            phase_scores[name] = reward
            satisfied.append(name)
            total += reward
        else:
            phase_scores[name] = 0.0

    for penalty in spec.get("penalties", []):
        name = penalty["name"]
        p = float(penalty.get("penalty", 0.0))
        if _eval_penalty(atoms, penalty):
            penalty_scores[name] = -p
            triggered_penalties.append(name)
            total -= p

    for bonus in spec.get("bonuses", []):
        name = bonus["name"]
        r = float(bonus.get("reward", 0.0))
        if _eval_bonus(atoms, bonus):
            bonus_scores[name] = r
            triggered_bonuses.append(name)
            total += r

    floor = float(spec.get("floor", 0.0))
    ceiling = float(spec.get("ceiling", 1.0))
    proc_score = max(floor, min(ceiling, total))

    return RewardResult(
        proc_score=round(proc_score, 4),
        phase_scores=phase_scores,
        penalties=penalty_scores,
        bonuses=bonus_scores,
        satisfied_phases=satisfied,
        triggered_penalties=triggered_penalties,
        triggered_bonuses=triggered_bonuses,
    )


__all__ = ["RewardResult", "load_spec", "score"]
