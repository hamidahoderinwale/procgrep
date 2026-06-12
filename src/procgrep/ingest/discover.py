"""Discover trajectory datasets on the Hugging Face Hub via metadata.

Instead of a hard-coded list, query the Hub for candidate trajectory datasets
and return their metadata (downloads, likes, tags, last-modified), ranked by
downloads. Feeds the curation catalog:

    discover() -> [DatasetMeta]            (Hub metadata, no download)
    plan(meta.id) -> adapter + confidence  (procgrep.ingest, cheap sniff)
    curate(...)   -> redundancy stats      (on demand, the expensive step)

The ``huggingface_hub`` library is an optional dependency, imported lazily.

Design decisions (benefit / price):

1. Query the Hub by keyword and author, ranked by downloads, rather than
   curating a fixed list. Benefit: new trajectory datasets surface without code
   changes. Price: keyword recall is imperfect, so a dataset that matches none
   of the query terms is missed (mitigated by a broad ``DEFAULT_QUERIES`` set
   plus an author crawl).
2. ``huggingface_hub`` is imported lazily and is an optional dependency.
   Benefit: the rest of procgrep installs and runs without it. Price: discovery
   fails only when actually called without the dependency, not at import time.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_QUERIES = (
    "trajectories",
    "traces",
    "rollouts",
    "agent trajectories",
    "agent traces",
    "agent rollouts",
    "swe-agent",
    "openhands",
    "mini-swe-agent",
    "swe-bench trajectories",
    "tool use trajectories",
    "react agent trajectories",
    "sft trajectories",
)

# Orgs known to publish coding-agent trace datasets; crawled in full so we catch
# datasets whose names don't contain a search keyword.
DEFAULT_AUTHORS = (
    "nebius",
    "nvidia",
    "SWE-bench",
    "SWE-Gym",
    "R2E-Gym",
    "Kwai-Klear",
    "all-hands",
    "princeton-nlp",
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
    authors: tuple[str, ...] = DEFAULT_AUTHORS,
    limit_per_query: int = 50,
    min_downloads: int = 0,
) -> list[DatasetMeta]:
    """Find trajectory datasets via keyword search + known-author crawl.

    Deduped by id and ranked by downloads. ``authors`` are crawled in full so
    datasets whose names lack a search keyword are still caught; relevance is
    decided downstream by the adapter sniff, not here.

    Raises:
        ImportError: ``huggingface_hub`` not installed.
    """
    try:
        from huggingface_hub import list_datasets
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "procgrep.discover requires `huggingface_hub` (ships with `datasets`)."
        ) from exc

    seen: dict[str, DatasetMeta] = {}

    def _add(d: object) -> None:
        ident = getattr(d, "id", None)
        if not ident or ident in seen:
            return
        seen[ident] = DatasetMeta(
            id=str(ident),
            downloads=int(getattr(d, "downloads", 0) or 0),
            likes=int(getattr(d, "likes", 0) or 0),
            last_modified=str(getattr(d, "last_modified", "") or ""),
            tags=tuple(getattr(d, "tags", []) or []),
        )

    for query in queries:
        for d in list_datasets(search=query, limit=limit_per_query):
            _add(d)
    for author in authors:
        for d in list_datasets(author=author, limit=limit_per_query):
            _add(d)

    metas = [m for m in seen.values() if m.downloads >= min_downloads]
    return sorted(metas, key=lambda m: m.downloads, reverse=True)


__all__ = ["DatasetMeta", "discover"]
