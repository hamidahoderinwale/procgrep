"""Task loading and the run directory's frozen instance list.

Intent: SWE-bench instances come from HF `datasets` once, at prepare time, and
are frozen into the run directory as `instances.jsonl`; every later step
(subprocess arms, re-measurement) reads the frozen file, never the network.
Read this when changing what a "task" is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INSTANCES_NAME = "instances.jsonl"


def load_swebench_instances(
    subset: str = "verified",
    split: str = "test",
    *,
    instance_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load SWE-bench instances from HF datasets (requires the `swebench` extra)."""
    from datasets import load_dataset  # heavy; only the prepare step needs it
    from minisweagent.run.benchmarks.swebench import DATASET_MAPPING

    dataset_path = DATASET_MAPPING.get(subset, subset)
    instances = sorted(load_dataset(dataset_path, split=split), key=lambda i: i["instance_id"])
    if instance_ids is not None:
        wanted = set(instance_ids)
        instances = [i for i in instances if i["instance_id"] in wanted]
        missing = wanted - {i["instance_id"] for i in instances}
        if missing:
            raise ValueError(f"instance ids not in {dataset_path}:{split}: {sorted(missing)}")
    if limit is not None:
        instances = instances[:limit]
    return [dict(i) for i in instances]


def write_instances_jsonl(path: Path, instances: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.write_text("".join(json.dumps(i) + "\n" for i in instances))
    return path


def read_instances_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
