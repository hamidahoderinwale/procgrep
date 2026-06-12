"""Live procedural monitoring of running agent trajectories.

The "production loop" version of the deployment-signal example. The
companion script `04_deployment_signal.py` walks one trajectory
prefix-by-prefix to demonstrate the mechanism; this one wraps that
mechanism in the shape you would actually run inline with a live
agent:

* watch multiple eager rules simultaneously (any rule whose
  violation is detectable on a partial prefix);
* iterate the bundled corpus and find every trajectory that fires
  at least one rule mid-stream;
* on the first eager violation, log the rule, print the step it
  fired at, compute how much of the trajectory's remaining budget
  was saved by interrupting, and break the loop.

Two optional flags help the script double as the recording for a
workshop-paper demo:

* ``--realtime`` paces output by sleeping ``--step-delay`` seconds
  per atom, so an asciinema capture shows the matcher firing in
  natural time instead of instant batch.
* ``--interrupt`` (default on) breaks the per-trajectory loop on the
  first eager violation; disable it to keep replaying past the
  violation, useful for ablations that compare with and without
  early-stop.

Run from the repository root:

    python examples/python/07_live_monitor.py
    python examples/python/07_live_monitor.py --realtime --step-delay 0.2
    python examples/python/07_live_monitor.py --no-interrupt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from procgrep import Trace, canonicalize, load_patterns, match_patterns
from procgrep.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "examples" / "synthetic_traces.jsonl"
RULES = ROOT / "examples" / "rules" / "stuck_edit_loop.yaml"


def select_eager_rules(patterns: list) -> frozenset[str]:
    """Pick rules that can fire meaningfully on a partial prefix.

    A `must_hold: false` rule fires when its forbidden pattern
    *appears*, which is monotone in prefix length: once it fires
    at step k, it stays fired. A `must_hold: true` rule fires
    when its required pattern is *absent*, which is the opposite
    — every short prefix trivially violates it before the
    required pattern has had a chance to appear. Watching the
    latter live produces a flood of spurious step-1 violations.
    The live monitor only watches the eager (must-hold-false) set.
    """
    return frozenset(r.name for r in patterns if not r.must_hold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="sleep between steps so the output paces like a live stream",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.15,
        help="seconds to sleep per atom when --realtime is set (default 0.15)",
    )
    parser.add_argument(
        "--interrupt",
        dest="interrupt",
        action="store_true",
        default=True,
        help="break the loop on first eager violation (default)",
    )
    parser.add_argument(
        "--no-interrupt",
        dest="interrupt",
        action="store_false",
        help="continue replaying past the first violation",
    )
    return parser.parse_args()


def stream(
    target: Trace,
    patterns,
    eager: frozenset[str],
    *,
    realtime: bool,
    step_delay: float,
    interrupt: bool,
) -> dict:
    """Replay one trajectory atom-by-atom and watch for eager violations.

    Returns a small dict summarizing what happened: which rule (if
    any) fired first, at which step, and how many atoms were saved
    by interrupting.
    """
    print(
        f"\n=== streaming {target.trace_id} (agent={target.agent}, length={len(target.atoms)}) ==="
    )
    fired_rule: str | None = None
    fired_at: int | None = None
    for k in range(1, len(target.atoms) + 1):
        prefix = Trace(
            trace_id=target.trace_id,
            agent=target.agent,
            atoms=target.atoms[:k],
            group=target.group,
            metadata=target.metadata,
        )
        report = match_patterns([prefix], patterns)
        firing = [v for v in report.violations.get(target.trace_id, []) if v in eager]
        atom = target.atoms[k - 1]
        if firing:
            marker = f"!! {','.join(firing)}"
            print(f"  step {k:3d}  {atom:15s}  {marker}")
            if fired_rule is None:
                fired_rule = firing[0]
                fired_at = k
            if interrupt:
                break
        else:
            print(f"  step {k:3d}  {atom:15s}")
        if realtime:
            time.sleep(step_delay)

    if fired_rule is None:
        print("  -> no eager violation; trajectory ran to completion")
        return {"fired_rule": None, "fired_at": None, "atoms_saved": 0}
    saved = len(target.atoms) - fired_at
    pct = 100.0 * saved / len(target.atoms)
    print(
        f"  -> INTERRUPT at step {fired_at}: rule {fired_rule!r}; saved {saved}/{len(target.atoms)} atoms ({pct:.0f}% of budget)"
    )
    return {"fired_rule": fired_rule, "fired_at": fired_at, "atoms_saved": saved}


def main() -> None:
    args = parse_args()
    traces = canonicalize(list(read_jsonl(TRACES)), adapter="swe-agent")
    patterns = load_patterns(RULES)
    eager = select_eager_rules(patterns)

    print(f"loaded {len(traces)} trajectories; watching eager rules: {sorted(eager)}")
    if args.realtime:
        print(f"realtime pacing enabled at {args.step_delay:.2f}s per step")
    if not args.interrupt:
        print("interrupt disabled: trajectories will replay through to completion")

    summaries: list[dict] = []
    for target in traces:
        result = stream(
            target,
            patterns,
            eager,
            realtime=args.realtime,
            step_delay=args.step_delay,
            interrupt=args.interrupt,
        )
        summaries.append({"trace_id": target.trace_id, **result})

    total = sum(s["atoms_saved"] for s in summaries)
    fired = [s for s in summaries if s["fired_rule"] is not None]
    print(
        f"\n{'=' * 60}\nsummary: {len(fired)}/{len(summaries)} trajectories interrupted; {total} total atoms saved"
    )
    for s in summaries:
        if s["fired_rule"] is not None:
            print(
                f"  {s['trace_id']:10s} step {s['fired_at']:3d}  rule={s['fired_rule']}  saved={s['atoms_saved']}"
            )


if __name__ == "__main__":
    main()
