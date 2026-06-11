"""ProcGrep explorer backend — query full agent-trajectory datasets, live.

Intent: a small FastAPI service that ingests a Hugging Face trajectory dataset
through procgrep, caches the canonicalized traces, and answers structural
queries (for example "an edit streak of five or more with no test") over the
whole dataset rather than a fixed 500-trace sample. Read this when changing the
explorer's server behavior or its caching.

Design decisions (benefit / price):
 1. Import procgrep; do not reimplement canonicalization or queries.
    Benefit: improvements to the library flow here for free, and the static
    essay and this live backend share one definition of a "procedure".
    Price: the Space pins a procgrep revision and must be rebuilt to pick up
    library changes.
 2. Cache canonicalized traces per dataset in memory under a small LRU.
    Benefit: the expensive step (ingest plus canonicalize) runs once, after
    which a query is a microsecond string scan.
    Price: the first query on a cold dataset pays the full ingest cost, and
    memory grows with the number of cached datasets (bounded by MAX_DATASETS).
 3. Bound each ingest to MAX_TRACES with a timeout.
    Benefit: predictable latency and memory on the free CPU tier.
    Price: a very large dataset is sampled to the cap, not scanned in full;
    the response reports when this happens.
 4. A query is a regular expression over the canonical atom spine, matching the
    static essay's query box exactly.
    Benefit: one query language across the paper and the live demo.
    Price: the spine drops argument-level detail, by design in procgrep.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from procgrep.ingest import ingest

# ── configuration ────────────────────────────────────────────────────────────
MAX_TRACES = 20000  # per-dataset ingest cap (design decision 3); raise as memory allows
MAX_DATASETS = 6  # cached datasets before LRU eviction (design decision 2)
INGEST_TIMEOUT_S = 60.0
HIT_SAMPLE = 50  # matched traces returned to the client
STATIC = Path(__file__).parent / "static"

# A short, curated starting set; the client may query any dataset id.
SUGGESTED = (
    "nebius/SWE-agent-trajectories",
    "ElenaFu/SWE-agent-trajectories",
    "nebius/SWE-rebench-openhands-trajectories",
    "SWE-bench/SWE-smith-trajectories",
)

# Canonical atom palette (kept in sync with the static explorer).
ATOM_COLOR = {
    "search_repo": "#585E53",
    "read_file": "#5692E5",
    "edit": "#CB4D20",
    "create_file": "#3D7AD8",
    "run_test": "#20A380",
    "submit": "#14110E",
    "think": "#b7b1a7",
    "localize": "#8C1040",
    "delete_file": "#A03D18",
    "error": "#B4184F",
    "other": "#d9d4cc",
}


@dataclass(frozen=True)
class CachedTrace:
    """One canonicalized trajectory, with its atom spine pre-joined for scans."""

    trace_id: str
    agent: str
    atoms: tuple[str, ...]
    spine: str  # " ".join(atoms) + " ", so `(edit ){5,}` style regexes match
    task: str = ""  # the task/instance the trajectory solved (trace.group), if known


# dataset id -> traces. OrderedDict gives us LRU order cheaply.
_CACHE: OrderedDict[str, list[CachedTrace]] = OrderedDict()
_META: dict[str, dict] = {}  # dataset id -> {adapter, n_traces, truncated, n_models}


def _load(dataset: str) -> list[CachedTrace]:
    """Return cached canonicalized traces for ``dataset``, ingesting on a miss.

    Ingest is bounded by MAX_TRACES and a timeout; results are cached under an
    LRU of size MAX_DATASETS (design decisions 2 and 3).
    """
    if dataset in _CACHE:
        _CACHE.move_to_end(dataset)
        return _CACHE[dataset]

    traces, plan = ingest(dataset, limit=MAX_TRACES, timeout=INGEST_TIMEOUT_S)
    cached = [
        CachedTrace(
            t.trace_id, t.agent, tuple(t.atoms), " ".join(t.atoms) + " ", str(t.group or "")
        )
        for t in traces
    ]
    _CACHE[dataset] = cached
    _META[dataset] = {
        "adapter": plan.adapter,
        "n_traces": len(cached),
        "truncated": len(cached) >= MAX_TRACES,
        **_stats(cached),
    }
    while len(_CACHE) > MAX_DATASETS:
        evicted, _ = _CACHE.popitem(last=False)
        _META.pop(evicted, None)
    return cached


def _stats(traces: list[CachedTrace]) -> dict[str, float | int]:
    """Quick dataset stats: behavioral diversity, conciseness, CoT length, dups.

    diversity_bits = mean per-trajectory action entropy (how varied each agent's
    actions are); median_len = median trace length (conciseness); median_cot =
    median count of `think` steps per trace (chain-of-thought length).
    """
    if not traces:
        return {"n_models": 0}
    lens = [len(t.atoms) for t in traces]
    thinks = [sum(a == "think" for a in t.atoms) for t in traces]
    ents: list[float] = []
    for t in traces:
        counts = Counter(t.atoms)
        n = len(t.atoms) or 1
        ents.append(-sum((k / n) * math.log2(k / n) for k in counts.values()))
    uniq = len({t.spine for t in traces})
    return {
        "diversity_bits": round(sum(ents) / len(ents), 2),
        "median_len": int(median(lens)),
        "median_cot": int(median(thinks)),
        "exact_dup_rate": round(1 - uniq / len(traces), 3),
        "n_models": len({t.agent for t in traces}),
    }


def _action_mix(traces: list[CachedTrace]) -> dict[str, float]:
    """Normalized frequency of each atom across the given traces."""
    counts: Counter[str] = Counter(a for t in traces for a in t.atoms)
    total = sum(counts.values()) or 1
    return {a: round(n / total, 4) for a, n in counts.most_common()}


# ── API ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="ProcGrep explorer", docs_url="/api")


class QueryRequest(BaseModel):
    dataset: str = SUGGESTED[0]
    pattern: str  # regex over the space-joined atom spine


@app.get("/datasets")
def datasets() -> JSONResponse:
    """Suggested datasets plus which ones are already warm in the cache."""
    return JSONResponse(
        {"suggested": list(SUGGESTED), "cached": list(_CACHE.keys()), "meta": _META}
    )


@app.post("/query")
def query(req: QueryRequest) -> JSONResponse:
    """Run a structural regex over a whole dataset's atom spines.

    Returns the match count, per-model match rates, the matched-vs-all action
    mix, and a sample of matched traces. Errors (bad pattern, ingest failure)
    come back as a JSON ``error`` rather than a 500 so the client can show them.
    """
    try:
        rx = re.compile(req.pattern)
    except re.error as exc:
        return JSONResponse({"error": f"invalid pattern: {exc}"}, status_code=200)
    try:
        traces = _load(req.dataset)
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=200)

    t0 = time.perf_counter()
    hits = [t for t in traces if rx.search(t.spine)]
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    by_model: dict[str, dict[str, int]] = {}
    for t in traces:
        by_model.setdefault(t.agent, {"n": 0, "hits": 0})["n"] += 1
    for t in hits:
        by_model[t.agent]["hits"] += 1
    models = sorted(
        (
            {"model": m, "rate": round(c["hits"] / c["n"], 4), "n": c["n"]}
            for m, c in by_model.items()
        ),
        key=lambda r: -r["rate"],
    )

    return JSONResponse(
        {
            "dataset": req.dataset,
            "pattern": req.pattern,
            "n_traces": len(traces),
            "n_hits": len(hits),
            "elapsed_ms": round(elapsed_ms, 2),
            "truncated": _META.get(req.dataset, {}).get("truncated", False),
            "stats": {
                k: _META.get(req.dataset, {}).get(k)
                for k in (
                    "diversity_bits",
                    "median_len",
                    "median_cot",
                    "exact_dup_rate",
                    "n_models",
                )
            },
            "by_model": models,
            "mix_all": _action_mix(traces),
            "mix_hits": _action_mix(hits) if hits else {},
            "atom_color": ATOM_COLOR,
            "hits": [
                {
                    "trace_id": t.trace_id,
                    "model": t.agent,
                    "task": t.task,
                    "atoms": list(t.atoms[:200]),
                }
                for t in hits[:HIT_SAMPLE]
            ],
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
