"""Match procedural patterns against canonical atom sequences.

Level 1 pattern matcher: regexes over the space-joined atom sequence.
Each rule's ``must_hold`` flag inverts the semantics — a sequence
fails when a ``must_hold=False`` rule matches or when a
``must_hold=True`` rule does not.

Rule files are YAML::

    rules:
      - name: no_long_edit_loops
        description: No run of 5+ consecutive edits.
        pattern: "(edit ){5,}"
        must_hold: false

Audit-to-monitor methodology
----------------------------

A defensible rule file is grounded in empirical evidence: each rule
should correspond to a pattern shown to correlate with a target
behavior on a labeled corpus.

1. Label a corpus (manual, held-out test gap, or LLM-judge).
2. Discover candidates via
   :func:`procgrep.stats.discriminative_procedures`.
3. Validate on a held-out split; record precision/recall per rule.
4. Deploy: load with :func:`load_patterns`, evaluate live trajectory
   prefixes with :func:`match_patterns`.

Failure-correlated is not hacking-correlated. Do not publish a rule
file claiming to detect behavior X without the validation above. See
``examples/rules/known_failure_patterns.yaml`` for source-cited
failure patterns from Beyond Resolution Rates, Code Agent Behaviour,
and HAI-Code.
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
        pattern: Regex evaluated against
            ``" ".join(trace.atoms) + " "``. The trailing space
            simplifies right-anchored matches on the last atom.
        must_hold: True -> trace fails when regex does NOT match.
            False -> trace fails when regex DOES match.
    """

    name: str
    description: str
    pattern: str
    must_hold: bool


@dataclass(frozen=True)
class PatternReport:
    """Aggregate result of matching patterns against traces.

    Attributes:
        violations: ``trace_id -> [rule_name, ...]`` for traces with
            at least one failure.
        pass_rate_per_rule: ``rule_name -> fraction of passing
            traces``.
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
    """Evaluate every pattern against every trace."""
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
    """Validate a YAML rule dict and build a `Pattern`."""
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
    """True iff the trace fails this rule."""
    matched = rx.search(encoded) is not None
    return (must_hold and not matched) or ((not must_hold) and matched)


__all__ = ["Pattern", "PatternReport", "load_patterns", "match_patterns"]
