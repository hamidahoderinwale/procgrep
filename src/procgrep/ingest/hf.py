"""HuggingFace dataset ingestion helper.

Converts a HuggingFace dataset directly into a list of canonical
:class:`procgrep.types.Trace` objects::

    parent = from_hf("SWE-bench/SWE-smith-trajectories", adapter="swe-smith",
                     split="tool", trace_id_field="traj_id", agent_field="model")
    diff = lineage_diff(parent=parent, child=child)

The ``datasets`` library is an optional dependency, imported lazily;
:func:`from_hf` raises a clear :class:`ImportError` when it's absent.
"""

from __future__ import annotations

from typing import Any

from procgrep.canonicalize import canonicalize
from procgrep.types import Trace


def from_hf(
    dataset_name: str,
    *,
    adapter: str,
    split: str | None = None,
    config_name: str | None = None,
    streaming: bool = False,
    trace_id_field: str = "trace_id",
    agent_field: str = "agent",
    group_field: str | None = "group",
    limit: int | None = None,
    revision: str | None = None,
) -> list[Trace]:
    """Load a HuggingFace dataset and canonicalize to Traces.

    Wraps :func:`datasets.load_dataset` and pipes rows through
    :func:`procgrep.canonicalize`. ``adapter`` must already be
    registered (built-ins self-register at import).

    Args:
        split: Pass an explicit split for predictable behavior. With
            ``None``, ``load_dataset`` returns a ``DatasetDict``.
        streaming: Stream rather than materialize; pair with ``limit``
            to bound memory.
        limit: Cap on rows ingested (honored in both modes).
        revision: Git revision on the Hub.

    Raises:
        ImportError: ``datasets`` not installed.
        KeyError: ``adapter`` not registered.
    """
    load_dataset = _import_load_dataset()

    ds = load_dataset(
        dataset_name,
        name=config_name,
        split=split,
        streaming=streaming,
        revision=revision,
    )

    rows = _bounded(ds, limit=limit, streaming=streaming)

    return canonicalize(
        rows,
        adapter=adapter,
        trace_id_field=trace_id_field,
        agent_field=agent_field,
        group_field=group_field,
    )


def _import_load_dataset() -> Any:
    """Lazily import :func:`datasets.load_dataset`.

    Split out so the type checker doesn't see the optional import and
    so the import-error path is unit-testable.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - import-time error path
        raise ImportError(
            "procgrep.hf.from_hf requires the optional `datasets` library. "
            "Install with: pip install datasets"
        ) from exc
    return load_dataset


def _bounded(
    ds: Any,
    *,
    limit: int | None,
    streaming: bool,
) -> list[dict[str, object]]:
    """Materialize at most ``limit`` rows from a HuggingFace dataset.

    ``ds`` is :class:`typing.Any` because the concrete
    ``Dataset`` / ``IterableDataset`` types aren't visible without
    ``datasets`` installed; we duck-type
    ``select`` / ``take`` / ``__iter__`` / ``__len__``.
    """
    if limit is None:
        return [dict(row) for row in ds]
    if streaming:
        return [dict(row) for row in ds.take(limit)]
    capped = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in capped]


__all__ = ["from_hf"]
