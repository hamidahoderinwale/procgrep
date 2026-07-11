"""Close the loop: a run directory's trace arms -> atoms -> `procgrep.verify`.

Intent: the deterministic measurement path. Given the trace files, everything
here is bit-reproducible: ingest via the mini-swe-agent adapter, a seeded BPE
vocabulary, `verify`, and a seeded paired bootstrap for uncertainty. Read this
when changing what a run's result means.

Design decisions:

1. Uncertainty lives here, not in the core. `verify` stays a deterministic
   point estimate; this module resamples paired instances and reports 95%
   percentile intervals around both axes. Benefit: core untouched. Price:
   the bootstrap re-runs `verify` n_boot times (cheap: milliseconds each).
2. Atoms are the model's *attempted* actions. The adapter reads assistant
   messages, so a guard-blocked command still appears as its attempted atom;
   `guard_blocked` counts in rows and metadata record what actually ran.
   Benefit: one canonical ingest path. Price: block-mode behavior shifts are
   visible only after the model adapts, not at the substitution itself.
3. A spec loaded from YAML has no target (the core drops it on save), so
   `measure_run` accepts winner-labeled traces and grafts a re-derived target
   via `ProcedureSpec.from_winners`. Benefit: specs stay portable. Price: the
   caller must keep the winner corpus around.
4. The vocabulary is fit over both arms plus the winner corpus, so all three
   share one segmentation. Fitting on the arms alone can hand the target a
   disjoint procedure support (BPE merges are corpus-greedy) and saturate
   both JSDs at 1.0, which reads as weak enforcement no matter what happened.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from procgrep.bpe import ProcedureVocabulary, fit_bpe
from procgrep.ingest.adapters.mini_swe_agent import mini_swe_agent_adapter
from procgrep.program import verify
from procgrep.reward import ProcedureSpec
from procgrep.types import Trace
from procgrep_runner.manifest import SPEC_NAME
from procgrep_runner.run import ARM_BASELINE, ARM_ENFORCED

MEASURE_SUMMARY_NAME = "measure.summary.json"
MEASURE_ROWS_NAME = "measure.rows.jsonl"


def arm_traces(
    run_dir: Path,
    arm: str,
    *,
    grades: dict[str, bool] | None = None,
) -> list[Trace]:
    """Load one arm's trajectories as procgrep `Trace`s.

    ``grades`` maps instance_id to resolved (from an external SWE-bench
    evaluation); ungraded traces count as unresolved, matching `verify`.
    """
    traces: list[Trace] = []
    for path in sorted((Path(run_dir) / "arms" / arm).glob("*.traj.json")):
        record = json.loads(path.read_text())
        stem = path.name[: -len(".traj.json")]
        instance_id, _, rep = stem.rpartition(".r")
        instance_id = record.get("instance_id") or instance_id
        guard = (record.get("procgrep_runner") or {}).get("guard") or {}
        traces.append(
            Trace(
                trace_id=f"{stem}.{arm}",
                agent=arm,
                atoms=list(mini_swe_agent_adapter(record)),
                group=arm,
                metadata={
                    "instance_id": instance_id,
                    "replicate": int(rep) if rep.isdigit() else 0,
                    "resolved": bool((grades or {}).get(instance_id, False)),
                    "exit_status": (record.get("info") or {}).get("exit_status"),
                    "guard_blocked": guard.get("blocked", 0),
                    "guard_steered": guard.get("steered", 0),
                },
            )
        )
    return traces


def spec_with_target(
    spec: ProcedureSpec,
    outcome_labeled_traces: list[Trace],
    vocab: ProcedureVocabulary,
    *,
    outcome_field: str = "resolved",
) -> ProcedureSpec:
    """Graft a target fingerprint re-derived from winner traces onto ``spec``.

    The core drops the target on YAML save on purpose (it is a runtime
    artifact); this restores it from a corpus whose metadata carries the
    boolean ``outcome_field``.
    """
    derived = ProcedureSpec.from_winners(outcome_labeled_traces, vocab, outcome_field=outcome_field)
    return replace(spec, target=derived.target)


def _percentile_ci(samples: list[float], *, level: float = 0.95) -> tuple[float, float]:
    ordered = sorted(samples)
    lo_q = (1.0 - level) / 2.0
    lo = ordered[round(lo_q * (len(ordered) - 1))]
    hi = ordered[round((1.0 - lo_q) * (len(ordered) - 1))]
    return (round(lo, 6), round(hi, 6))


def paired_bootstrap(
    before: list[Trace],
    after: list[Trace],
    spec: ProcedureSpec,
    vocab: ProcedureVocabulary,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    outcome_field: str = "resolved",
) -> dict[str, Any]:
    """Resample paired instances; 95% CIs for both `verify` axes.

    ``jsd_move_toward_target`` is before-JSD minus after-JSD: positive means
    the enforced arm's fingerprint sits closer to the spec target.
    """
    by_arm: dict[str, dict[str, list[Trace]]] = {"b": {}, "a": {}}
    for key, traces in (("b", before), ("a", after)):
        for t in traces:
            by_arm[key].setdefault(str(t.metadata.get("instance_id")), []).append(t)
    ids = sorted(set(by_arm["b"]) & set(by_arm["a"]))
    if len(ids) < 2:
        return {
            "skipped": f"need >=2 paired instances, have {len(ids)}",
            "n_paired_instances": len(ids),
        }

    rng = random.Random(seed)
    jsd_moves: list[float] = []
    outcome_deltas: list[float] = []
    for _ in range(n_boot):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        resampled_before = [t for iid in sample for t in by_arm["b"][iid]]
        resampled_after = [t for iid in sample for t in by_arm["a"][iid]]
        report = verify(resampled_before, resampled_after, spec, vocab, outcome_field=outcome_field)
        before_jsd, after_jsd = report.fingerprint_jsd_to_target
        jsd_moves.append(before_jsd - after_jsd)
        outcome_deltas.append(report.outcome_delta)
    return {
        "n_paired_instances": len(ids),
        "n_boot": n_boot,
        "seed": seed,
        "jsd_move_toward_target_ci95": _percentile_ci(jsd_moves),
        "outcome_delta_ci95": _percentile_ci(outcome_deltas),
    }


def measure_run(
    run_dir: Path,
    *,
    spec: ProcedureSpec | None = None,
    vocab: ProcedureVocabulary | None = None,
    vocab_size: int = 64,
    vocab_seed: int = 0,
    winners: list[Trace] | None = None,
    grades: dict[str, dict[str, bool]] | None = None,
    outcome_field: str = "resolved",
    jsd_improvement_eps: float = 1e-3,
    n_boot: int = 1000,
    boot_seed: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    """Measure a paired run: `verify` verdict plus bootstrap CIs, written to disk.

    ``grades`` maps arm name to {instance_id: resolved}; without it the
    outcome axis is reported but flagged ``graded: false`` (all traces count
    unresolved, so read only the behavior axis). ``winners`` supplies the
    target when the run's saved spec has none.
    """
    run_dir = Path(run_dir)
    grades = grades or {}
    before = arm_traces(run_dir, ARM_BASELINE, grades=grades.get(ARM_BASELINE))
    after = arm_traces(run_dir, ARM_ENFORCED, grades=grades.get(ARM_ENFORCED))
    if not before or not after:
        raise ValueError(
            f"missing traces under {run_dir}/arms (before={len(before)}, after={len(after)})"
        )

    spec = spec or ProcedureSpec.from_yaml(run_dir / SPEC_NAME)
    fit_corpus = [*before, *after, *(winners or [])]
    vocab = vocab or fit_bpe([t.atoms for t in fit_corpus], vocab_size=vocab_size, seed=vocab_seed)
    if spec.target is None:
        if winners is None:
            raise ValueError(
                "spec has no target fingerprint (YAML drops it); pass winners= "
                "traces labeled with the outcome field to re-derive one"
            )
        spec = spec_with_target(spec, winners, vocab, outcome_field=outcome_field)

    report = verify(
        before,
        after,
        spec,
        vocab,
        outcome_field=outcome_field,
        jsd_improvement_eps=jsd_improvement_eps,
    )
    summary = {
        "run_id": run_dir.name,
        "verify": asdict(report),
        "graded": bool(grades),
        "n": {"before": len(before), "after": len(after)},
        "guard_blocked_total": sum(t.metadata.get("guard_blocked") or 0 for t in after),
        "guard_steered_total": sum(t.metadata.get("guard_steered") or 0 for t in after),
        "bootstrap": paired_bootstrap(
            before, after, spec, vocab, n_boot=n_boot, seed=boot_seed, outcome_field=outcome_field
        ),
        "vocab": {"size": vocab.size, "fit_size": vocab_size, "seed": vocab_seed},
        "jsd_improvement_eps": jsd_improvement_eps,
    }
    if write:
        (run_dir / MEASURE_SUMMARY_NAME).write_text(json.dumps(summary, indent=2))
        with (run_dir / MEASURE_ROWS_NAME).open("w") as fh:
            for trace in [*before, *after]:
                fh.write(
                    json.dumps(
                        {
                            "trace_id": trace.trace_id,
                            "arm": trace.agent,
                            "n_atoms": len(trace.atoms),
                            "atoms": trace.atoms,
                            **trace.metadata,
                        }
                    )
                    + "\n"
                )
    return summary
