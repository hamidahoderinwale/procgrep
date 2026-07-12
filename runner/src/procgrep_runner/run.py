"""Paired-arm execution: baseline vs enforced over the same tasks and replicates.

Intent: turn (spec, instances, config) into a sealed run directory with two
trace arms that `measure` can hand to `procgrep.verify`. Read this when
changing what a run produces or how arms differ.

Design decisions:

1. One agent class for both arms. The baseline arm is a `GuardedAgent` with
   no guard and an untouched prompt; the enforced arm differs only by the
   enforcement mode. Benefit: any behavioral difference is the enforcement,
   not the harness. Price: none found yet.
2. Instances run in their own subprocess by default (``isolate=True``). A
   stalled sandbox or a poisoned client cannot cascade across the batch; a
   crash becomes a row with an ``exit_status``, never a lost batch.
   Benefit: batch survives per-task failure. Price: process startup cost.
3. Run directories are never overwritten (`prepare_run` fails on an existing
   directory); artifacts are append-only (rows.jsonl).
4. ``seed`` is recorded in the manifest but not fed to the model: hosted
   models do not honor sampling seeds. It exists so paired arms and future
   re-runs are labeled comparably.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import signal
import subprocess
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.utils.serialize import UNSET, recursive_merge

from procgrep.guard import OnViolation, ProcedureGuard
from procgrep.reward import ProcedureSpec
from procgrep_runner.harness import GuardedAgent
from procgrep_runner.manifest import SPEC_NAME, RunManifest, build_manifest, write_manifest
from procgrep_runner.tasks import INSTANCES_NAME, write_instances_jsonl

ARM_BASELINE = "baseline"
ARM_ENFORCED = "enforced"
ARMS = (ARM_BASELINE, ARM_ENFORCED)
CONFIG_NAME = "config.json"
ROWS_NAME = "rows.jsonl"
DEFAULT_SWEBENCH_CONFIG = builtin_config_dir / "benchmarks" / "swebench.yaml"

InstanceRunner = Callable[[Path, str, str, int], dict]


def build_config(
    *,
    config_specs: list[str | Path] | None = None,
    model_name: str | None = None,
    model_class: str | None = None,
    environment_class: str | None = None,
    cost_limit: float | None = None,
) -> dict[str, Any]:
    """Merge mini-swe-agent's swebench config with user specs and overrides."""
    specs = [str(DEFAULT_SWEBENCH_CONFIG)] + [str(s) for s in (config_specs or [])]
    configs = [get_config_from_spec(s) for s in specs]
    configs.append(
        {
            "environment": {"environment_class": environment_class or UNSET},
            "model": {"model_name": model_name or UNSET, "model_class": model_class or UNSET},
            "agent": {} if cost_limit is None else {"cost_limit": cost_limit},
        }
    )
    return recursive_merge(*configs)


def prepare_run(
    run_dir: Path,
    *,
    spec: ProcedureSpec,
    instances: list[dict[str, Any]],
    config: dict[str, Any],
    mode: str,
    on_violation: OnViolation = "block",
    replicates: int = 1,
    seed: int = 0,
    subset: str | None = None,
    split: str | None = None,
) -> RunManifest:
    """Freeze a run's inputs into ``run_dir`` and seal the manifest.

    Fails if ``run_dir`` already exists: prior runs are never overwritten.
    """
    if mode not in ("prompt", "guard"):
        raise ValueError(f"unknown enforcement mode {mode!r}; expected prompt or guard")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    spec.to_yaml(run_dir / SPEC_NAME)
    write_instances_jsonl(run_dir / INSTANCES_NAME, instances)
    (run_dir / CONFIG_NAME).write_text(json.dumps(config, indent=2))
    model_cfg = config.get("model") or {}
    manifest = build_manifest(
        run_dir,
        run_id=run_dir.name,
        spec_name=spec.name or "unnamed",
        mode=mode,
        on_violation=on_violation if mode == "guard" else None,
        arms=ARMS,
        model_name=model_cfg.get("model_name"),
        model_class=model_cfg.get("model_class"),
        environment_class=(config.get("environment") or {}).get("environment_class"),
        subset=subset,
        split=split,
        instance_ids=tuple(i["instance_id"] for i in instances),
        replicates=replicates,
        seed=seed,
        cost_limit=(config.get("agent") or {}).get("cost_limit"),
    )
    write_manifest(run_dir, manifest)
    return manifest


def _default_model_factory(config: dict[str, Any]):
    from minisweagent.models import get_model

    return get_model(config=config.get("model", {}))


def _default_env_factory(config: dict[str, Any], instance: dict[str, Any]):
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    return get_sb_environment(config, instance)


def _enforcement_prompt(spec: ProcedureSpec) -> str:
    # {% raw %} so rule prose can never be parsed as jinja by the scaffold's
    # StrictUndefined template rendering.
    return "\n\nProcedure directive:\n{% raw %}" + spec.to_prompt() + "{% endraw %}"


def traj_path(run_dir: Path, arm: str, instance_id: str, replicate: int) -> Path:
    return Path(run_dir) / "arms" / arm / f"{instance_id}.r{replicate}.traj.json"


def run_instance(
    instance: dict[str, Any],
    *,
    spec: ProcedureSpec,
    arm: str,
    mode: str,
    config: dict[str, Any],
    run_dir: Path,
    on_violation: OnViolation = "block",
    replicate: int = 0,
    model_factory: Callable[[dict], Any] | None = None,
    env_factory: Callable[[dict, dict], Any] | None = None,
) -> dict[str, Any]:
    """Execute one (instance, arm, replicate); always leave a trajectory file.

    Exceptions are recorded (``exit_status`` = exception class name, traceback
    in the trajectory's info) and returned as a row, never raised: the caller
    is a batch that must survive per-task failure.
    """
    config = copy.deepcopy(config)
    instance_id = instance["instance_id"]
    path = traj_path(run_dir, arm, instance_id, replicate)
    path.parent.mkdir(parents=True, exist_ok=True)

    agent_kwargs = dict(config.get("agent", {}))
    agent_kwargs["output_path"] = path
    guard: ProcedureGuard | None = None
    if arm == ARM_ENFORCED and mode == "guard":
        guard = ProcedureGuard(spec, on_violation=on_violation)
    if arm == ARM_ENFORCED and mode == "prompt":
        agent_kwargs["system_template"] = agent_kwargs.get(
            "system_template", ""
        ) + _enforcement_prompt(spec)

    model = (model_factory or _default_model_factory)(config)
    run_meta = {
        "arm": arm,
        "mode": mode if arm == ARM_ENFORCED else "none",
        "on_violation": on_violation if guard is not None else None,
        "replicate": replicate,
        "spec": spec.name,
        "repo": instance.get("repo"),
        "difficulty": instance.get("difficulty"),
        "model_name": getattr(model.config, "model_name", None),
    }

    exit_status: str | None = None
    submission = ""
    extra_info: dict[str, Any] = {}
    agent = None
    try:
        env = (env_factory or _default_env_factory)(config, instance)
        agent = GuardedAgent(model, env, guard=guard, **agent_kwargs)
        info = agent.run(instance["problem_statement"])
        exit_status = info.get("exit_status")
        submission = info.get("submission", "") or ""
    except Exception as exc:
        exit_status = type(exc).__name__
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(exc)}
    finally:
        record_info = {"exit_status": exit_status, "submission": submission, **extra_info}
        if agent is not None:
            agent.save(
                path, {"instance_id": instance_id, "runner_run": run_meta, "info": record_info}
            )
        else:
            # Environment construction failed before an agent existed; leave a
            # stub so the arm directory still reflects the attempt.
            path.write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "runner_run": run_meta,
                        "info": record_info,
                        "messages": [],
                    },
                    indent=2,
                )
            )
        if submission:
            from minisweagent.run.benchmarks.swebench import update_preds_file

            update_preds_file(
                path.parent / f"preds.r{replicate}.json",
                instance_id,
                getattr(model.config, "model_name", "unknown"),
                submission,
            )

    return {
        "instance_id": instance_id,
        "arm": arm,
        "replicate": replicate,
        "exit_status": exit_status,
        "submitted": bool(submission),
        "guard_blocked": (agent.serialize()["procgrep_runner"].get("guard") or {}).get("blocked")
        if agent is not None and guard is not None
        else None,
        "traj_path": str(path),
    }


def _subprocess_argv(run_dir: Path, instance_id: str, arm: str, replicate: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "procgrep_runner.cli",
        "instance",
        "--run-dir",
        str(run_dir),
        "--instance-id",
        instance_id,
        "--arm",
        arm,
        "--replicate",
        str(replicate),
    ]


def run_instance_subprocess(
    run_dir: Path,
    instance_id: str,
    arm: str,
    replicate: int,
    *,
    timeout: float | None = None,
    argv_builder: Callable[[Path, str, str, int], list[str]] = _subprocess_argv,
) -> dict[str, Any]:
    """Run one instance in a child process; the child prints its row as JSON.

    Any failure mode (crash, timeout, garbled output) degrades to a row with a
    runner ``exit_status`` so the batch keeps going. The child runs in its own
    process group and the whole group is killed on timeout: sandbox tunnel
    descendants inherit the output pipes, and killing only the child leaves
    the pipe read blocked forever (observed as a 25h mid-batch hang).
    """
    base = {"instance_id": instance_id, "arm": arm, "replicate": replicate, "submitted": False}
    proc = subprocess.Popen(
        argv_builder(run_dir, instance_id, arm, replicate),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.communicate(timeout=30)
        return {**base, "exit_status": "RunnerTimeout"}
    if proc.returncode != 0:
        return {
            **base,
            "exit_status": "RunnerSubprocessError",
            "returncode": proc.returncode,
            "stderr_tail": stderr[-2000:],
        }
    for line in reversed(stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {**base, "exit_status": "RunnerProtocolError", "stdout_tail": stdout[-2000:]}


def run_paired(
    *,
    spec: ProcedureSpec,
    instances: list[dict[str, Any]],
    config: dict[str, Any],
    run_dir: Path,
    mode: str = "guard",
    on_violation: OnViolation = "block",
    replicates: int = 1,
    seed: int = 0,
    subset: str | None = None,
    split: str | None = None,
    isolate: bool = True,
    instance_timeout: float | None = 7200,
    instance_runner: InstanceRunner | None = None,
    model_factory: Callable[[dict], Any] | None = None,
    env_factory: Callable[[dict, dict], Any] | None = None,
) -> list[dict[str, Any]]:
    """Run every (instance, replicate) through both arms; append rows.jsonl.

    ``instance_runner`` overrides how one (instance, arm, replicate) executes
    (tests inject fakes here). Otherwise ``isolate=True`` shells out per
    instance for crash isolation and ``isolate=False`` runs in-process
    (required when injecting ``model_factory`` / ``env_factory``).
    """
    run_dir = Path(run_dir)
    prepare_run(
        run_dir,
        spec=spec,
        instances=instances,
        config=config,
        mode=mode,
        on_violation=on_violation,
        replicates=replicates,
        seed=seed,
        subset=subset,
        split=split,
    )
    rows_path = run_dir / ROWS_NAME
    rows: list[dict[str, Any]] = []
    for instance in instances:
        for replicate in range(replicates):
            for arm in ARMS:
                iid = instance["instance_id"]
                try:
                    if instance_runner is not None:
                        row = instance_runner(run_dir, iid, arm, replicate)
                    elif isolate:
                        row = run_instance_subprocess(
                            run_dir, iid, arm, replicate, timeout=instance_timeout
                        )
                    else:
                        row = run_instance(
                            instance,
                            spec=spec,
                            arm=arm,
                            mode=mode,
                            config=config,
                            run_dir=run_dir,
                            on_violation=on_violation,
                            replicate=replicate,
                            model_factory=model_factory,
                            env_factory=env_factory,
                        )
                except Exception as exc:  # a single task must never kill the batch
                    row = {
                        "instance_id": iid,
                        "arm": arm,
                        "replicate": replicate,
                        "exit_status": type(exc).__name__,
                        "submitted": False,
                    }
                rows.append(row)
                with rows_path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
    return rows
