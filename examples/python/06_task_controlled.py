"""Task-controlled procedural comparison.

Two analyses on the same fixture, both holding `instance_id` fixed:

(a) Cross-agent JSD at fixed task. Removes task heterogeneity as a
    confound of cross-agent comparisons. Asks: given identical input,
    do two agents produce structurally different procedures?

(b) Success-vs-failure JSD at fixed task. Pairs a resolved trajectory
    with an unresolved trajectory on the same instance. Asks: at
    fixed task, does outcome carry a procedural signature?

The data-prep pattern is the load-time preprocessor described in
docs/STUDIES.md study #3: read the raw records, run `canonicalize` once,
then rebuild the `group` label per analysis from `metadata`. The
shared BPE vocabulary is fit once over the whole corpus so the
vocabulary itself is not a partition-specific confound.

Run from the repository root:

    python examples/python/06_task_controlled.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from procgrep import (
    ProcedureVocabulary,
    Trace,
    canonicalize,
    discriminative_procedures,
    encode,
    fit_bpe,
    jsd,
    jsd_matrix,
)
from procgrep.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "examples" / "data" / "synthetic_task_traces.jsonl"


def mean_pairwise_jsd(
    traces: list[Trace], vocab: ProcedureVocabulary, label_a: str, label_b: str
) -> float | None:
    """Mean pairwise JSD between every (a, b) cross-pair under the given grouping.

    Returns None if either group has zero members.
    """
    fps = encode(traces, vocab=vocab)
    a = [fp for fp in fps if fp.group == label_a]
    b = [fp for fp in fps if fp.group == label_b]
    if not a or not b:
        return None
    pairs = [jsd(x.distribution(), y.distribution()) for x in a for y in b]
    return sum(pairs) / len(pairs)


def main() -> None:
    raw = list(read_jsonl(TRACES))
    traces = canonicalize(raw, adapter="swe-agent", group_field=None)
    print(f"loaded {len(traces)} traces")
    instances = sorted({str(t.metadata["instance_id"]) for t in traces})
    agents = sorted({t.agent for t in traces})
    outcomes = sorted({str(t.metadata["outcome"]) for t in traces})
    print(f"  instances : {instances}")
    print(f"  agents    : {agents}")
    print(f"  outcomes  : {outcomes}\n")

    # One shared vocabulary fit on the whole corpus. The vocabulary
    # is the bridge between every grouping below; refitting per
    # partition would make later comparisons incommensurable.
    vocab = fit_bpe((t.atoms for t in traces), vocab_size=20, seed=0)
    print(
        f"shared vocabulary: {vocab.size} tokens ({len(vocab.atoms)} atoms + {len(vocab.merges)} merges)\n"
    )

    # (a) cross-agent at fixed task
    print("=" * 64)
    print("(a) cross-agent JSD at fixed instance_id")
    print("=" * 64)
    per_instance_jsd: dict[str, float] = {}
    for inst in instances:
        inst_traces = [t for t in traces if t.metadata["instance_id"] == inst]
        labeled = [replace(t, group=t.agent) for t in inst_traces]
        per_agent_jsd: list[float] = []
        for agent_a, agent_b in combinations(agents, 2):
            value = mean_pairwise_jsd(labeled, vocab, agent_a, agent_b)
            if value is not None:
                per_agent_jsd.append(value)
                print(f"  {inst:24s} {agent_a:10s} vs {agent_b:10s} JSD = {value:.4f}")
        if per_agent_jsd:
            per_instance_jsd[inst] = sum(per_agent_jsd) / len(per_agent_jsd)
    if per_instance_jsd:
        mean_across = sum(per_instance_jsd.values()) / len(per_instance_jsd)
        print(f"\n  mean cross-agent JSD averaged across instances: {mean_across:.4f}")

    # Compare against the unconditional cross-agent JSD (no task control)
    # to see how much task heterogeneity contributes.
    pooled = [replace(t, group=t.agent) for t in traces]
    pooled_fps = encode(pooled, vocab=vocab)
    pooled_matrix = jsd_matrix(pooled_fps, group_by="group")
    print("\n  unconditional cross-agent JSD (pooled across instances):")
    for record in pooled_matrix.to_records():
        if str(record["row"]) < str(record["col"]):
            print(f"    {record['row']:10s} vs {record['col']:10s} JSD = {record['jsd']:.4f}")

    # (b) success-vs-failure at fixed task
    print("\n" + "=" * 64)
    print("(b) success-vs-failure JSD at fixed instance_id")
    print("=" * 64)
    per_instance_outcome_jsd: dict[str, float] = {}
    for inst in instances:
        inst_traces = [t for t in traces if t.metadata["instance_id"] == inst]
        labeled = [replace(t, group=str(t.metadata["outcome"])) for t in inst_traces]
        value = mean_pairwise_jsd(labeled, vocab, "resolved", "unresolved")
        if value is None:
            print(f"  {inst:24s} skipped (need both outcomes in this instance)")
            continue
        per_instance_outcome_jsd[inst] = value
        print(f"  {inst:24s} resolved vs unresolved  JSD = {value:.4f}")
    if per_instance_outcome_jsd:
        mean_across = sum(per_instance_outcome_jsd.values()) / len(per_instance_outcome_jsd)
        print(f"\n  mean success-vs-failure JSD averaged across instances: {mean_across:.4f}")

    # Top procedures separating resolved from unresolved, pooled across
    # instances. With task held implicitly through pairing per instance,
    # this is "what procedures differentiate outcome at fixed task" rolled
    # up across the corpus.
    by_outcome = [replace(t, group=str(t.metadata["outcome"])) for t in traces]
    by_outcome_fps = encode(by_outcome, vocab=vocab)
    print("\n  top discriminative procedures: resolved vs unresolved")
    print(f"  {'procedure':24s} {'p_resolved':>12s} {'p_unresolved':>14s} {'log_odds':>10s}")
    top = discriminative_procedures(
        by_outcome_fps,
        vocab,
        group_a="resolved",
        group_b="unresolved",
        k=5,
        ranking="log_odds",
        group_by="group",
    )
    for m in top:
        print(f"  {m.procedure:24s} {m.p_a:12.3f} {m.p_b:14.3f} {m.log_odds:10.3f}")

    # Sanity: per-(instance, outcome) trace count, so the reader can
    # see the pairing structure underlying the JSD numbers.
    print("\n" + "=" * 64)
    print("trace counts per (instance_id, outcome)")
    print("=" * 64)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for t in traces:
        counts[(str(t.metadata["instance_id"]), str(t.metadata["outcome"]))] += 1
    for (inst, outcome), n in sorted(counts.items()):
        print(f"  {inst:24s} {outcome:12s} n = {n}")


if __name__ == "__main__":
    main()
