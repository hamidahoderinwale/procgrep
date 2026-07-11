"""Sealed run manifest: everything needed to re-measure a run.

Intent: one JSON that binds a paired run to its inputs, the spec (by content
hash), scaffold and model identity, task list, and seeds, so a later reader
can tell exactly what produced a trace directory. Read this when auditing a
run or checking whether two runs are comparable.

Design decisions:

1. The spec is stored as YAML inside the run directory and referenced by
   sha256. Benefit: the manifest stays small and the spec stays diffable.
   Price: two files must travel together (`read_manifest` verifies the hash).
2. Sampling is declared, not guaranteed: the manifest records model and
   sampling identity so a re-run is comparable, while the bit-reproducible
   part is the measurement path (traces -> atoms -> verify).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

MANIFEST_NAME = "manifest.json"
SPEC_NAME = "spec.yaml"


@dataclass(frozen=True)
class RunManifest:
    """Identity of one paired run; see the module docstring for the contract."""

    run_id: str
    created_at: str
    spec_name: str
    spec_sha256: str
    mode: str
    on_violation: str | None
    arms: tuple[str, ...]
    model_name: str | None
    model_class: str | None
    environment_class: str | None
    subset: str | None
    split: str | None
    instance_ids: tuple[str, ...]
    replicates: int
    seed: int
    cost_limit: float | None
    procgrep_version: str
    mini_swe_agent_version: str
    runner_version: str


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


def sha256_path(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(
    run_dir: Path,
    *,
    run_id: str,
    spec_name: str,
    mode: str,
    on_violation: str | None,
    arms: tuple[str, ...],
    model_name: str | None,
    model_class: str | None,
    environment_class: str | None,
    subset: str | None,
    split: str | None,
    instance_ids: tuple[str, ...],
    replicates: int,
    seed: int,
    cost_limit: float | None,
) -> RunManifest:
    """Build the manifest for a prepared run directory (spec.yaml must exist)."""
    return RunManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        spec_name=spec_name,
        spec_sha256=sha256_path(run_dir / SPEC_NAME),
        mode=mode,
        on_violation=on_violation,
        arms=arms,
        model_name=model_name,
        model_class=model_class,
        environment_class=environment_class,
        subset=subset,
        split=split,
        instance_ids=instance_ids,
        replicates=replicates,
        seed=seed,
        cost_limit=cost_limit,
        procgrep_version=_version("procgrep"),
        mini_swe_agent_version=_version("mini-swe-agent"),
        runner_version=_version("procgrep-runner"),
    )


def write_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    path = Path(run_dir) / MANIFEST_NAME
    path.write_text(json.dumps(asdict(manifest), indent=2))
    return path


def read_manifest(run_dir: Path, *, verify_spec_hash: bool = True) -> RunManifest:
    """Load a run's manifest; by default re-hash spec.yaml and fail on mismatch."""
    run_dir = Path(run_dir)
    raw = json.loads((run_dir / MANIFEST_NAME).read_text())
    raw["arms"] = tuple(raw["arms"])
    raw["instance_ids"] = tuple(raw["instance_ids"])
    manifest = RunManifest(**raw)
    if verify_spec_hash:
        actual = sha256_path(run_dir / SPEC_NAME)
        if actual != manifest.spec_sha256:
            raise ValueError(
                f"spec.yaml hash mismatch in {run_dir}: manifest says "
                f"{manifest.spec_sha256[:12]}, file is {actual[:12]}"
            )
    return manifest
