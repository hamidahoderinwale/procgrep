"""Deployment-signal use case: pattern matching on procedural prefixes.

Simulates a streaming agent run by replaying one of the bundled
synthetic traces atom-by-atom. After each new atom, the current
prefix is checked against a rule set; when a rule fires, the step
is flagged.

In production this would run inline with the live agent: the prefix
matcher decides whether to interrupt the run before the agent burns
more budget on a trajectory headed for a known failure shape.

Run from the repository root:

    python examples/python/04_deployment_signal.py
"""

from __future__ import annotations

from pathlib import Path

from procgrep import Trace, canonicalize, load_patterns, match_patterns
from procgrep.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "examples" / "data" / "synthetic_traces.jsonl"
RULES = ROOT / "examples" / "rules" / "stuck_edit_loop.yaml"

# Which rules to alert on at the prefix level. Some rules (such as
# `ends_in_submit` or `tests_run_at_least_once`) are properties of
# completed trajectories and would always fail mid-stream; we only
# watch eagerly for rules that can fire on a partial prefix.
EAGER_RULES = {"no_long_edit_loops"}


def main() -> None:
    traces = canonicalize(list(read_jsonl(TRACES)), adapter="swe-agent")
    patterns = load_patterns(RULES)

    target = next(t for t in traces if t.trace_id == "syn-004")
    print(f"streaming trajectory {target.trace_id}")
    print(f"  agent={target.agent}, group={target.group}")
    print(f"  atoms={target.atoms}")
    print(f"  length={len(target.atoms)}\n")

    print(f"{'step':>4s} {'atom':>15s}  {'eager violations':30s}")
    fired_at = None
    for k in range(1, len(target.atoms) + 1):
        prefix = Trace(
            trace_id=target.trace_id,
            agent=target.agent,
            atoms=target.atoms[:k],
            group=target.group,
            metadata=target.metadata,
        )
        report = match_patterns([prefix], patterns)
        violations = report.violations.get(target.trace_id, [])
        eager = [v for v in violations if v in EAGER_RULES]
        marker = " ".join(eager) if eager else ""
        print(f"  {k:4d} {target.atoms[k - 1]:15s}  {marker}")
        if eager and fired_at is None:
            fired_at = k

    if fired_at is None:
        print("\nno eager rule fired during the trajectory")
    else:
        print(
            f"\nfirst eager rule fired at step {fired_at}; "
            "in production this would trigger an interrupt"
        )


if __name__ == "__main__":
    main()
