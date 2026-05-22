"""HuggingFace dataset ingestion helper.

Thin wrapper that converts a HuggingFace dataset directly into a list of
canonical :class:`procgrep.types.Trace` objects, so the common workflow
collapses to a few lines::

    from procgrep import lineage_diff
    from procgrep.hf import from_hf

    parent = from_hf("SWE-bench/SWE-smith-trajectories", adapter="swe-smith",
                     split="tool", trace_id_field="traj_id", agent_field="model")
    child = from_hf("your-org/post-trained-trajectories", adapter="swe-smith",
                    trace_id_field="traj_id", agent_field="model")
    diff = lineage_diff(parent=parent, child=child)

The ``datasets`` library is an *optional* dependency; this module imports
it lazily so installing procgrep without ``datasets`` keeps the package
importable. ``from_hf`` raises a clear :class:`ImportError` if the
library is missing.
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
    """Load a HuggingFace dataset and canonicalize to a list of Traces.

    Wraps :func:`datasets.load_dataset` and feeds the rows through
    :func:`procgrep.canonicalize`. The named ``adapter`` must already be
    registered (built-in adapters self-register at package import time).

    Args:
        dataset_name: HuggingFace dataset identifier
            (e.g., ``"SWE-bench/SWE-smith-trajectories"``).
        adapter: Adapter name (registered via
            :func:`procgrep.canonicalize.register_adapter`). The adapter
            is responsible for translating each row's scaffold-specific
            shape into a canonical atom sequence.
        split: Optional split name (``"train"``, ``"validation"``,
            ``"tool"``, etc.). If ``None``, ``load_dataset`` returns a
            :class:`DatasetDict` and this function iterates whatever
            default split is associated with the first key -- for
            predictable behavior, pass an explicit split.
        config_name: Optional dataset configuration name, forwarded as
            ``name=`` to ``load_dataset``.
        streaming: If ``True``, stream the dataset rather than
            materializing it. Useful for very large corpora; pair with
            ``limit`` to bound memory.
        trace_id_field: Per-row key holding the trace id. Passed to
            :func:`canonicalize`.
        agent_field: Per-row key holding the agent name. Passed to
            :func:`canonicalize`.
        group_field: Optional per-row key holding a grouping label.
            Passed to :func:`canonicalize`.
        limit: Optional cap on the number of rows ingested. Honored
            both in eager and streaming modes; recommended during
            development to keep iteration tight.
        revision: Optional git revision (commit hash or branch) of the
            dataset on the Hub.

    Returns:
        Canonical :class:`Trace` objects in dataset row order.

    Raises:
        ImportError: If the optional ``datasets`` library is not
            installed. Install with ``pip install datasets``.
        KeyError: If the requested adapter is not registered. Propagated
            from :func:`procgrep.canonicalize.get_adapter`.
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
    """Lazily import :func:`datasets.load_dataset` with a clear error.

    Split out so the type checker doesn't have to model the optional
    ``datasets`` import, and so the import error path is unit-testable.
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

    ``ds`` is typed as :class:`typing.Any` because the concrete
    ``datasets.Dataset`` / ``IterableDataset`` types are not statically
    visible without the optional dependency installed; the helper relies
    on the duck-typed ``select`` / ``take`` / ``__iter__`` / ``__len__``
    interfaces that both expose.
    """
    if limit is None:
        return [dict(row) for row in ds]
    if streaming:
        return [dict(row) for row in ds.take(limit)]
    # Eager mode: use index slicing to avoid iterating past the cap.
    capped = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in capped]


__all__ = ["from_hf"]
