"""Match each task to the agent whose past procedure fits best.

The question this script answers, in plain language:

    If we had picked, for each task, the agent whose past behavior
    most resembles "what works on this kind of task," would we
    have solved more tasks than if we had just picked the best-
    overall agent every time, or picked at random?

The mechanism is deliberately simple:

1. For each task class (here, each `instance_id`), compute a
   *reference fingerprint*: the mean procedure distribution across
   every trajectory on that task that ended in `resolved`.
2. For each agent, compute the agent's own *signature
   fingerprint*: the mean procedure distribution across all of that
   agent's trajectories.
3. To pick an agent for a held-out task, find the agent whose
   signature is closest (smallest Jensen-Shannon divergence) to
   the held-out task's reference fingerprint.
4. Score: did that agent actually resolve the held-out task on
   that instance in the data?

Baselines compared against:

* **best-overall**: always pick the agent with the highest resolve
  rate across all tasks.
* **random**: pick a uniformly random agent for each task,
  averaged across seeds.

What the script reports:

* The number of tasks each strategy resolves, out of total.
* The matching strategy's pick per task, with whether it
  succeeded.

The script needs traces whose metadata carries `instance_id` and
`outcome`. It runs on the bundled
``examples/synthetic_task_traces.jsonl`` fixture as a smoke test;
pass ``--traces`` to point at a real corpus (e.g. a procgrep-
canonicalized export of an 84-agent SWE-bench leaderboard).

Run from the repository root:

    python examples/python/09_match_agent_to_task.py
    python examples/python/09_match_agent_to_task.py --traces my_traces.jsonl
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from procgrep import (
    Fingerprint,
    Trace,
    canonicalize,
    encode,
    fit_bpe,
    jsd,
)
from procgrep.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = ROOT / "examples" / "synthetic_task_traces.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES,
        help="JSONL trace file (default: bundled task-controlled fixture)",
    )
    parser.add_argument(
        "--adapter",
        default="swe-agent",
        help="canonicalize adapter (default: swe-agent)",
    )
    parser.add_argument(
        "--task-key",
        default="instance_id",
        help="metadata field holding the task identifier (default: instance_id)",
    )
    parser.add_argument(
        "--outcome-key",
        default="outcome",
        help="metadata field holding the resolved/unresolved label (default: outcome)",
    )
    parser.add_argument(
        "--resolved-value",
        default="resolved",
        help="value of --outcome-key that means 'task was solved' (default: resolved)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=50, help="BPE target vocabulary size (default: 50)"
    )
    parser.add_argument(
        "--random-seeds",
        type=int,
        default=200,
        help="seeds to average the random-baseline over (default: 200)",
    )
    return parser.parse_args()


def require_metadata(traces: list[Trace], task_key: str, outcome_key: str) -> None:
    missing_task = sum(1 for t in traces if t.metadata.get(task_key) is None)
    missing_outcome = sum(1 for t in traces if t.metadata.get(outcome_key) is None)
    if missing_task or missing_outcome:
        raise SystemExit(
            f"corpus is missing required metadata: "
            f"{missing_task} traces lack '{task_key}', "
            f"{missing_outcome} traces lack '{outcome_key}'. "
            f"Use --task-key / --outcome-key to point at the correct fields, "
            f"or supply a corpus that carries them."
        )


def mean_distribution(fps: list[Fingerprint]) -> np.ndarray:
    if not fps:
        return np.zeros(0)
    return np.mean([fp.distribution() for fp in fps], axis=0)


def pick_agent_by_fit(task_reference: np.ndarray, agent_signatures: dict[str, np.ndarray]) -> str:
    """Pick the agent whose signature is closest (lowest JSD) to the task reference."""
    return min(agent_signatures, key=lambda a: jsd(agent_signatures[a], task_reference))


def main() -> None:
    args = parse_args()
    raw = list(read_jsonl(args.traces))
    traces = canonicalize(raw, adapter=args.adapter)
    require_metadata(traces, args.task_key, args.outcome_key)

    vocab = fit_bpe((t.atoms for t in traces), vocab_size=args.vocab_size, seed=0)
    fps = encode(traces, vocab=vocab)
    fp_by_id = {fp.trace_id: fp for fp in fps}

    # Index: trace -> (agent, task, outcome)
    info: dict[str, dict] = {}
    for t in traces:
        info[t.trace_id] = {
            "agent": t.agent,
            "task": str(t.metadata[args.task_key]),
            "outcome": str(t.metadata[args.outcome_key]),
        }

    agents = sorted({i["agent"] for i in info.values()})
    tasks = sorted({i["task"] for i in info.values()})

    print(f"loaded {len(traces)} traces; {len(agents)} agents x {len(tasks)} tasks")
    print(f"  agents: {agents}")
    print(f"  tasks : {tasks}")

    # Did each (agent, task) cell actually resolve?
    cell_resolved: dict[tuple[str, str], bool] = {}
    for meta in info.values():
        cell = (meta["agent"], meta["task"])
        if meta["outcome"] == args.resolved_value:
            cell_resolved[cell] = True
        elif cell not in cell_resolved:
            cell_resolved[cell] = False

    # Best-overall: agent with highest total resolve count.
    resolves_per_agent = {
        a: sum(1 for (ag, _), r in cell_resolved.items() if ag == a and r) for a in agents
    }
    best_overall_agent = max(resolves_per_agent, key=lambda a: resolves_per_agent[a])

    # evaluation
    print("\nleave-one-task-out evaluation:")
    print(f"  {'task':24s} {'best-fit':12s} {'fit?':>4s}  {'best-overall':12s} {'b-o?':>4s}")
    fit_solves = 0
    best_overall_solves = 0

    for held_out in tasks:
        # Reference fingerprint for this task class: mean of every
        # resolved trajectory on this exact task in the data.
        # In a real leaderboard this would be "resolved trajectories
        # on tasks of similar class" -- here task class = instance id
        # because the synthetic fixture has one task per class.
        ref_fps = [
            fp_by_id[tid]
            for tid, meta in info.items()
            if meta["task"] == held_out and meta["outcome"] == args.resolved_value
        ]
        if not ref_fps:
            # No resolved trajectory exists on this task in the data;
            # the fit-based picker has nothing to align to. Skip.
            continue
        reference = mean_distribution(ref_fps)

        # Agent signatures from OTHER tasks (leave-one-task-out).
        agent_sigs: dict[str, np.ndarray] = {}
        for a in agents:
            other_fps = [
                fp_by_id[tid]
                for tid, meta in info.items()
                if meta["agent"] == a and meta["task"] != held_out
            ]
            if other_fps:
                agent_sigs[a] = mean_distribution(other_fps)
        if not agent_sigs:
            continue

        fit_pick = pick_agent_by_fit(reference, agent_sigs)
        fit_ok = cell_resolved.get((fit_pick, held_out), False)
        bo_ok = cell_resolved.get((best_overall_agent, held_out), False)
        fit_solves += int(fit_ok)
        best_overall_solves += int(bo_ok)
        print(
            f"  {held_out:24s} {fit_pick:12s} {'OK' if fit_ok else '--':>4s}"
            f"  {best_overall_agent:12s} {'OK' if bo_ok else '--':>4s}"
        )

    # Random baseline (Monte Carlo).
    rng = random.Random(0)
    random_total = 0
    n_evaluated = 0
    for held_out in tasks:
        ref_fps = [
            fp_by_id[tid]
            for tid, meta in info.items()
            if meta["task"] == held_out and meta["outcome"] == args.resolved_value
        ]
        if not ref_fps:
            continue
        n_evaluated += 1
        for _ in range(args.random_seeds):
            pick = rng.choice(agents)
            if cell_resolved.get((pick, held_out), False):
                random_total += 1
    random_mean = random_total / args.random_seeds if args.random_seeds else 0.0

    # summary
    print(f"\nsummary (over {n_evaluated} held-out tasks):")
    print(
        f"  match-by-fit         resolves {fit_solves}/{n_evaluated}  ({100 * fit_solves / max(n_evaluated, 1):.0f}%)"
    )
    print(
        f"  best-overall ({best_overall_agent})  resolves {best_overall_solves}/{n_evaluated}  "
        f"({100 * best_overall_solves / max(n_evaluated, 1):.0f}%)"
    )
    print(
        f"  random  (mean over {args.random_seeds} seeds)  resolves {random_mean:.2f}/{n_evaluated}  ({100 * random_mean / max(n_evaluated, 1):.0f}%)"
    )
    if fit_solves > best_overall_solves:
        diff = fit_solves - best_overall_solves
        print(
            f"\n  fit-matching beats best-overall by {diff} task{'s' if diff != 1 else ''} on this corpus."
        )
    elif fit_solves < best_overall_solves:
        diff = best_overall_solves - fit_solves
        print(
            f"\n  best-overall beats fit-matching by {diff} task{'s' if diff != 1 else ''} on this corpus."
        )
    else:
        print("\n  fit-matching and best-overall tie on this corpus.")
    print(
        "\n  caveat: small fixtures yield small absolute differences; "
        "the meaningful version of this comparison runs on a real corpus "
        "(say, the 84-agent leaderboard) where the resolve counts are large."
    )


if __name__ == "__main__":
    main()
