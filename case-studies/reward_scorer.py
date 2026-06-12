"""Procedural reward scorer for agent trajectories.

Reads a procgrep reward spec (YAML) and scores a trajectory against it,
returning a structured partial-reward signal in addition to binary pass/fail.

Usage:
    python reward_scorer.py --spec examples/reward_spec_swe_agent.yaml \
                            --fingerprint results/fingerprints_child_n500.jsonl \
                            --instance django__django-12345

    python reward_scorer.py --spec examples/reward_spec_swe_agent.yaml \
                            --fingerprint results/fingerprints_child_n500.jsonl \
                            --all --top 20

Outputs a JSON reward record per trajectory:
    {
      "instance_id": "...",
      "binary_pass": true/false/null,
      "proc_score": 0.65,
      "phase_scores": {"exploration": 0.10, ...},
      "penalties": {"edit_streak": -0.15},
      "bonuses": {"test_driven": 0.10},
      "satisfied_phases": ["exploration", "diagnosis", "implementation"],
      "triggered_penalties": ["edit_streak"],
      "triggered_bonuses": [],
    }
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise SystemExit("pip install pyyaml")

HERE = Path(__file__).parent


# ── Pattern matching helpers ──────────────────────────────────────────────────

def has_atom(atoms: list[str], atom: str, min_count: int = 1) -> bool:
    return atoms.count(atom) >= min_count


def has_any_atom(atoms: list[str], atom_list: list[str],
                  min_occurrences: int = 1) -> bool:
    return any(atoms.count(a) >= min_occurrences for a in atom_list)


def has_contiguous_pattern(atoms: list[str], pattern: list[str]) -> bool:
    """Return True if pattern appears as a contiguous subsequence."""
    n, p = len(atoms), len(pattern)
    return any(atoms[i:i+p] == pattern for i in range(n - p + 1))


def has_sequence_within_gap(atoms: list[str], seq: list[str],
                             max_gap: int = 999) -> bool:
    """True if seq[0] appears, then seq[1] within max_gap steps after it."""
    if len(seq) != 2:
        # For longer sequences, check each consecutive pair
        for a, b in zip(seq[:-1], seq[1:]):
            if not has_sequence_within_gap(atoms, [a, b], max_gap):
                return False
        return True
    a, b = seq
    for i, atom in enumerate(atoms):
        if atom == a:
            window = atoms[i+1: i+1+max_gap]
            if b in window:
                return True
    return False


def first_occurrence(atoms: list[str], atom: str) -> int:
    """Index of first occurrence of atom, or len(atoms) if absent."""
    try:
        return atoms.index(atom)
    except ValueError:
        return len(atoms)


def atoms_before_first(atoms: list[str], target: str) -> list[str]:
    idx = first_occurrence(atoms, target)
    return atoms[:idx]


# ── Phase / penalty / bonus evaluation ───────────────────────────────────────

def eval_phase(atoms: list[str], phase: dict) -> bool:
    """Return True if this phase's conditions are satisfied."""
    before_first = phase.get("before_first")
    scope = atoms_before_first(atoms, before_first) if before_first else atoms

    if "require_any" in phase:
        req = phase["require_any"]
        min_occ = phase.get("min_occurrences", 1)
        atom_names = [r["atom"] for r in req if "atom" in r]
        if not has_any_atom(scope, atom_names, min_occ):
            return False

    if "require_pattern" in phase:
        pat = phase["require_pattern"]
        if not any(has_atom(scope, a) for a in pat):
            return False

    if "require_sequence" in phase:
        seq = phase["require_sequence"]
        max_gap = phase.get("max_gap", 999)
        min_occ = phase.get("min_occurrences", 1)
        found = 0
        for i in range(len(atoms)):
            if has_sequence_within_gap(atoms[i:], seq, max_gap):
                found += 1
                if found >= min_occ:
                    break
        if found < min_occ:
            return False

    if "require_absent_before" in phase:
        required_absent = phase["require_absent_before"]
        before_first_edit = atoms_before_first(atoms, phase.get("before_first", "edit"))
        if any(a in before_first_edit for a in required_absent):
            return False

    return True


def eval_penalty(atoms: list[str], penalty: dict) -> bool:
    """Return True if this penalty is triggered."""
    if "pattern" in penalty and penalty.get("contiguous", False):
        return has_contiguous_pattern(atoms, penalty["pattern"])
    if "require_absent_before" in penalty:
        required_absent = penalty["require_absent_before"]
        target = penalty.get("before_first", "edit")
        before = atoms_before_first(atoms, target)
        return not any(a in before for a in required_absent)
    return False


def eval_bonus(atoms: list[str], bonus: dict) -> bool:
    """Return True if this bonus is earned."""
    if "require_sequence" in bonus:
        seq = bonus["require_sequence"]
        before_first = bonus.get("before_first")
        scope = atoms_before_first(atoms, before_first) if before_first else atoms
        max_gap = bonus.get("max_gap", 999)
        return has_sequence_within_gap(scope, seq, max_gap)
    return False


# ── Main scorer ───────────────────────────────────────────────────────────────

def score(atoms: list[str], spec: dict) -> dict:
    """Score a canonical atom sequence against a reward spec."""
    phase_scores: dict[str, float] = {}
    penalty_scores: dict[str, float] = {}
    bonus_scores: dict[str, float] = {}
    satisfied: list[str] = []
    triggered_penalties: list[str] = []
    triggered_bonuses: list[str] = []

    total = 0.0

    for phase in spec.get("phases", []):
        name = phase["name"]
        reward = phase.get("reward", 0.0)
        if eval_phase(atoms, phase):
            phase_scores[name] = reward
            satisfied.append(name)
            total += reward
        else:
            phase_scores[name] = 0.0

    for penalty in spec.get("penalties", []):
        name = penalty["name"]
        p = penalty.get("penalty", 0.0)
        if eval_penalty(atoms, penalty):
            penalty_scores[name] = -p
            triggered_penalties.append(name)
            total -= p

    for bonus in spec.get("bonuses", []):
        name = bonus["name"]
        r = bonus.get("reward", 0.0)
        if eval_bonus(atoms, bonus):
            bonus_scores[name] = r
            triggered_bonuses.append(name)
            total += r

    floor = spec.get("floor", 0.0)
    ceiling = spec.get("ceiling", 1.0)
    proc_score = max(floor, min(ceiling, total))

    return {
        "proc_score": round(proc_score, 4),
        "phase_scores": phase_scores,
        "penalties": penalty_scores,
        "bonuses": bonus_scores,
        "satisfied_phases": satisfied,
        "triggered_penalties": triggered_penalties,
        "triggered_bonuses": triggered_bonuses,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Path to reward spec YAML")
    ap.add_argument("--fingerprint", required=True,
                    help="Path to fingerprint JSONL")
    ap.add_argument("--instance", default=None,
                    help="Score one specific instance_id")
    ap.add_argument("--all", action="store_true",
                    help="Score all instances and print summary")
    ap.add_argument("--top", type=int, default=10,
                    help="With --all: show top N and bottom N by proc_score")
    ap.add_argument("--layer", default="canonical",
                    help="atoms_canonical or atoms_native")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text())
    rows = [json.loads(l)
            for l in Path(args.fingerprint).read_text().splitlines()
            if l.strip()]

    results = []
    for row in rows:
        atoms = row.get(f"atoms_{args.layer}", [])
        if not atoms:
            continue
        r = score(atoms, spec)
        r["instance_id"] = row["instance_id"]
        r["binary_pass"] = row.get("resolved")
        results.append(r)

    if args.instance:
        match = [r for r in results if r["instance_id"] == args.instance]
        if match:
            print(json.dumps(match[0], indent=2))
        else:
            print(f"Not found: {args.instance}")
        return

    if args.all or not args.instance:
        scores = [r["proc_score"] for r in results]
        import numpy as np
        arr = np.array(scores)
        pass_results = [r for r in results if r.get("binary_pass") is True]
        fail_results = [r for r in results if r.get("binary_pass") is False]

        print(f"Spec: {spec['name']}")
        print(f"N={len(results)}  mean={arr.mean():.3f}  "
              f"median={np.median(arr):.3f}  std={arr.std():.3f}")
        if pass_results:
            ps = np.mean([r["proc_score"] for r in pass_results])
            print(f"  Pass trajectories (n={len(pass_results)}): "
                  f"mean proc_score={ps:.3f}")
        if fail_results:
            fs = np.mean([r["proc_score"] for r in fail_results])
            print(f"  Fail trajectories (n={len(fail_results)}): "
                  f"mean proc_score={fs:.3f}")

        # Phase hit rates
        print("\nPhase completion rates:")
        for phase in spec.get("phases", []):
            n_hit = sum(1 for r in results if phase["name"] in r["satisfied_phases"])
            print(f"  {phase['name']:25s}  {n_hit/len(results):>6.1%}  "
                  f"({n_hit}/{len(results)})")

        print("\nPenalty trigger rates:")
        for penalty in spec.get("penalties", []):
            n_hit = sum(1 for r in results
                        if penalty["name"] in r["triggered_penalties"])
            print(f"  {penalty['name']:25s}  {n_hit/len(results):>6.1%}  "
                  f"({n_hit}/{len(results)})")

        # Top and bottom N
        sorted_r = sorted(results, key=lambda x: -x["proc_score"])
        print(f"\nTop {args.top} proc_score:")
        for r in sorted_r[:args.top]:
            print(f"  {r['instance_id']:40s}  {r['proc_score']:.3f}  "
                  f"pass={r['binary_pass']}  "
                  f"phases={r['satisfied_phases']}")
        print(f"\nBottom {args.top} proc_score:")
        for r in sorted_r[-args.top:]:
            print(f"  {r['instance_id']:40s}  {r['proc_score']:.3f}  "
                  f"pass={r['binary_pass']}  "
                  f"penalties={r['triggered_penalties']}")


if __name__ == "__main__":
    main()
