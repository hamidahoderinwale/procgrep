"""Run agents under a `ProcedureSpec` and feed the traces back to `verify`.

Intent: the runner the procgrep README's roadmap promised, kept outside the
core on purpose. procgrep emits enforcement artifacts and measures traces;
this package is the scaffold-side host that executes paired baseline/enforced
arms (via mini-swe-agent) in sandboxes, seals a run manifest, and closes the
loop back to `procgrep.verify`. Read this when wiring an enforcement
experiment end to end.

Design decisions:

1. Paired arms are the unit of execution: every run produces a baseline arm
   and an enforced arm over the same tasks, replicates, and seeds, because
   `verify` consumes a before/after pair and pairing is what makes small
   samples usable. Benefit: per-task comparison. Price: 2x the runs.
2. The reproducibility contract is the measurement path, not the sampling.
   Model outputs are not reproducible even at temperature 0; what is
   bit-reproducible is traces -> atoms -> verify, and the manifest pins
   everything that path needs. Benefit: honest claim. Price: identical
   manifests can still produce different traces.
3. The core is never imported by procgrep; this package depends on procgrep
   plus mini-swe-agent. Benefit: procgrep stays model-free. Price: guard
   internals are reached through their public `ProcedureGuard` wrapper only.
"""

from procgrep_runner.harness import GuardedAgent
from procgrep_runner.manifest import RunManifest, read_manifest, sha256_path, write_manifest
from procgrep_runner.measure import arm_traces, measure_run, spec_with_target
from procgrep_runner.run import (
    ARM_BASELINE,
    ARM_ENFORCED,
    build_config,
    prepare_run,
    run_instance,
    run_paired,
)
from procgrep_runner.tasks import read_instances_jsonl, write_instances_jsonl

__all__ = [
    "ARM_BASELINE",
    "ARM_ENFORCED",
    "GuardedAgent",
    "RunManifest",
    "arm_traces",
    "build_config",
    "measure_run",
    "prepare_run",
    "read_instances_jsonl",
    "read_manifest",
    "run_instance",
    "run_paired",
    "sha256_path",
    "spec_with_target",
    "write_instances_jsonl",
    "write_manifest",
]
