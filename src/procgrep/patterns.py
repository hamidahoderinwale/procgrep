"""Match procedural patterns against canonical atom sequences.

This is the Level 1 pattern matcher: regular expressions over the
space-separated string form of each trajectory's atom sequence. Each
rule has a ``must_hold`` flag; the matcher reports the rule as a
violation when the regex matches a sequence that must NOT contain it,
or when the regex fails to match a sequence that MUST contain it.

The compositional invariant DSL (procedural-DSPy) with temporal
operators, soft predicates, and distribution-level invariants is
future work and is deliberately out of scope here. The current
matcher is intentionally limited so that the surface stays small
and obvious.

Rule files are YAML, of the form:

    rules:
      - name: no_long_edit_loops
        description: No run of 5+ consecutive edits.
        pattern: "(edit ){5,}"
        must_hold: false
      - name: localize_before_edit
        description: A localize must precede the first edit.
        pattern: "localize .* edit"
        must_hold: true
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from procgrep.types import Trace


@dataclass(frozen=True)
class Pattern:
    """One pattern-matching rule.

    Attributes:
        name: Stable identifier for the rule.
        description: Human-readable explanation.
        pattern: Python regular expression evaluated against the
            sequence ``" ".join(trace.atoms) + " "`` (trailing space
            simplifies right-anchored matches against the last atom).
        must_hold: If True, a trace fails when the regex does NOT
            match. If False, a trace fails when the regex DOES match.
    """

    name: str
    description: str
    pattern: str
    must_hold: bool


@dataclass(frozen=True)
class PatternReport:
    """Aggregate result of matching a set of patterns against traces.

    Attributes:
        patterns: The rules that were evaluated.
        violations: ``trace_id -> [rule_name, ...]`` for traces that
            failed at least one rule.
        pass_rate_per_rule: ``rule_name -> fraction of traces that
            satisfied the rule``.
    """

    patterns: tuple[Pattern, ...]
    violations: dict[str, list[str]]
    pass_rate_per_rule: dict[str, float]


def load_patterns(path: Path | str) -> list[Pattern]:
    """Load patterns from a YAML rules file."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "rules" not in raw:
        raise ValueError(f"expected top-level 'rules' key in {path}")
    rules_raw = raw["rules"]
    if not isinstance(rules_raw, list):
        raise TypeError(f"'rules' must be a list, got {type(rules_raw).__name__}")
    return [_pattern_from_dict(entry, source=str(path)) for entry in rules_raw]


def match_patterns(
    traces: Iterable[Trace],
    patterns: Iterable[Pattern],
) -> PatternReport:
    """Evaluate every pattern against every trace.

    Returns:
        A `PatternReport` summarizing per-trace violations and per-
        rule pass rates.
    """
    pattern_tuple = tuple(patterns)
    compiled = [(p, re.compile(p.pattern)) for p in pattern_tuple]
    trace_list = list(traces)

    violations: dict[str, list[str]] = {}
    failure_counts: dict[str, int] = {p.name: 0 for p in pattern_tuple}

    for trace in trace_list:
        encoded = " ".join(trace.atoms) + " "
        failures = [p.name for p, rx in compiled if _is_violation(rx, encoded, p.must_hold)]
        if failures:
            violations[trace.trace_id] = failures
            for name in failures:
                failure_counts[name] += 1

    n = max(len(trace_list), 1)
    pass_rate = {name: 1.0 - failure_counts[name] / n for name in failure_counts}

    return PatternReport(
        patterns=pattern_tuple,
        violations=violations,
        pass_rate_per_rule=pass_rate,
    )


def _pattern_from_dict(entry: Any, *, source: str) -> Pattern:
    """Validate a YAML rule dict and turn it into a `Pattern`."""
    if not isinstance(entry, dict):
        raise TypeError(f"rule entry in {source} is not a mapping: {entry!r}")
    missing = {"name", "pattern", "must_hold"} - set(entry)
    if missing:
        raise ValueError(f"rule in {source} missing keys: {sorted(missing)}")
    return Pattern(
        name=str(entry["name"]),
        description=str(entry.get("description", "")),
        pattern=str(entry["pattern"]),
        must_hold=bool(entry["must_hold"]),
    )


def _is_violation(rx: re.Pattern[str], encoded: str, must_hold: bool) -> bool:
    """Return True iff the trace fails this rule."""
    matched = rx.search(encoded) is not None
    return (must_hold and not matched) or ((not must_hold) and matched)


__all__ = ["Pattern", "PatternReport", "load_patterns", "match_patterns"]
