"""Discover trajectory datasets on the Hugging Face Hub via metadata.

Instead of a hard-coded list, query the Hub for candidate trajectory datasets
and return their metadata (downloads, likes, tags, last-modified), ranked by
downloads. Feeds the curation catalog:

    discover() -> [DatasetMeta]            (Hub metadata, no download)
    plan(meta.id) -> adapter + confidence  (procgrep.ingest, cheap sniff)
    curate(...)   -> redundancy stats      (on demand, the expensive step)

The ``huggingface_hub`` library is an optional dependency, imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_QUERIES = (
    "trajectories",
    "swe-agent trajectories",
    "openhands trajectories",
    "agent trajectories",
)


@dataclass(frozen=True)
class DatasetMeta:
    """Hub metadata for one candidate dataset."""

    id: str
    downloads: int
    likes: int
    last_modified: str
    tags: tuple[str, ...]


def discover(
    queries: tuple[str, ...] = DEFAULT_QUERIES,
    *,
    limit_per_query: int = 50,
    min_downloads: int = 0,
) -> list[DatasetMeta]:
    """Search the Hub for trajectory datasets, deduped and ranked by downloads.

    Raises:
        ImportError: ``huggingface_hub`` not installed.
    """
    try:
        from huggingface_hub import list_datasets  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "procgrep.discover requires `huggingface_hub` (ships with `datasets`)."
        ) from exc

    seen: dict[str, DatasetMeta] = {}
    for query in queries:
        for d in list_datasets(search=query, limit=limit_per_query):
            if d.id in seen:
                continue
            seen[d.id] = DatasetMeta(
                id=d.id,
                downloads=int(getattr(d, "downloads", 0) or 0),
                likes=int(getattr(d, "likes", 0) or 0),
                last_modified=str(getattr(d, "last_modified", "") or ""),
                tags=tuple(getattr(d, "tags", []) or []),
            )
    metas = [m for m in seen.values() if m.downloads >= min_downloads]
    return sorted(metas, key=lambda m: m.downloads, reverse=True)


__all__ = ["DatasetMeta", "discover"]
