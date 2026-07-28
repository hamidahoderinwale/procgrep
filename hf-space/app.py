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
 5. Prefer a precomputed spine store (HF dataset midah/procgrep-spines) over
    live ingest, falling back to live ingest when the store is missing a dataset
    or cannot be reached.
    Benefit: warm datasets answer instantly with no per-query streaming or
    canonicalization, and coverage can grow on CI rather than at request time.
    Price: store-backed datasets are only as fresh as the last refresh build;
    the weekly action keeps them current.
 6. The comparator (/compare) reuses the library's lineage primitives
    (fit_bpe, encode, discriminative_procedures, jsd) rather than reinventing
    them, and bounds each side to CMP_SAMPLE traces for the BPE/diff pass.
    Benefit: the side-by-side diff means exactly what the paper's lineage_diff
    means, and the heavy step stays bounded for interactive latency.
    Price: the discriminative-procedure ranking sees a sample, not all traces,
    on very large groups; the rendered trail stack is a further CMP_STACK cap.
 7. Cache /compare results by their key, and pre-warm the default landing pair
    in a background thread at startup.
    Benefit: the comparator's landing view answers instantly instead of paying
    the first BPE pass live; repeat toggles are free.
    Price: a bounded amount of memory for the cache, and the warm-up does one
    BPE pass at startup off the request path.
"""

from __future__ import annotations

import contextlib
import math
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from procgrep import (
    Trace,
    discriminative_procedures,
    encode,
    fit_bpe,
    jsd,
)
from procgrep.ingest import ingest
from procgrep.types import PROCEDURE_SEPARATOR

# Configuration.
MAX_TRACES = 20000  # per-dataset ingest cap (design decision 3); raise as memory allows
MAX_DATASETS = 6  # cached datasets before LRU eviction (design decision 2)
INGEST_TIMEOUT_S = 60.0
HIT_SAMPLE = 50  # matched traces returned to the client
STATIC = Path(__file__).parent / "static"
SPINE_REPO = "midah/procgrep-spines"  # precomputed store (design decision 5)
SPINE_FILE = "procgrep_spines.parquet"
CMP_SAMPLE = 1500  # traces per side fed to BPE + discriminative_procedures (design decision 6)
CMP_STACK = 30  # trails rendered per side in the comparator
CMP_VOCAB = 200  # BPE procedure-vocabulary size for the diff
CMP_TRAIL_CAP = 140  # atoms per rendered trail (fits the column at 3px/cell)
COMPARE_CACHE_MAX = 24  # cached /compare results before LRU eviction (design decision 7)

# A short, curated starting set; the client may query any dataset id.
SUGGESTED = (
    "nebius/SWE-agent-trajectories",
    "ElenaFu/SWE-agent-trajectories",
    "nebius/SWE-rebench-openhands-trajectories",
    "SWE-bench/SWE-smith-trajectories",
    "nvidia/Nemotron-SFT-Agentic-v2",
    "AlienKevin/SWE-ZERO-12M-trajectories",
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
    outcome: str = ""  # "resolved"/"unresolved"/"" from a genuine resolution label, when present
    cot_tokens: int = 0  # approx tokens the model generated (reasoning volume), chars/4


# dataset id -> traces. OrderedDict gives us LRU order cheaply.
_CACHE: OrderedDict[str, list[CachedTrace]] = OrderedDict()
_META: dict[str, dict] = {}  # dataset id -> {adapter, n_traces, truncated, n_models}

# Precomputed spine store, loaded once and shared (design decision 5). None until
# the first load attempt; an empty dict means "tried, nothing usable" so every
# dataset falls through to live ingest.
_STORE: dict[str, list[CachedTrace]] | None = None

# Cached /compare payloads keyed by (axis, dataset, left, right) (design decision 7).
_COMPARE_CACHE: OrderedDict[tuple[str, str, str, str], dict] = OrderedDict()


def _load_store() -> dict[str, list[CachedTrace]]:
    """Download and parse the precomputed spine store, grouped by dataset.

    Reconstructs atoms from the space-joined spine (lossless: atoms carry no
    internal spaces). Any failure (no repo, offline, bad file) yields an empty
    store so callers transparently fall back to live ingest.
    """
    global _STORE
    if _STORE is not None:
        return _STORE
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(SPINE_REPO, SPINE_FILE, repo_type="dataset")
        df = pd.read_parquet(path)
        store: dict[str, list[CachedTrace]] = {}
        for row in df.itertuples(index=False):
            spine = str(row.spine)
            atoms = tuple(spine.split())
            # outcome + cot_tokens are optional: older stores lack them, read defensively.
            outcome = str(getattr(row, "outcome", "") or "")
            cot = int(getattr(row, "cot_tokens", 0) or 0)
            store.setdefault(str(row.dataset), []).append(
                CachedTrace(
                    str(row.trace_id), str(row.agent), atoms, spine, str(row.task), outcome, cot
                )
            )
        _STORE = store
    except Exception:
        _STORE = {}
    return _STORE


def _load(dataset: str) -> list[CachedTrace]:
    """Return canonicalized traces for ``dataset``.

    Prefers the precomputed spine store (design decision 5); on a store miss,
    ingests live, bounded by MAX_TRACES and a timeout and cached under an LRU of
    size MAX_DATASETS (design decisions 2 and 3).
    """
    if dataset in _CACHE:
        _CACHE.move_to_end(dataset)
        return _CACHE[dataset]

    store = _load_store()
    if dataset in store:
        cached = store[dataset]
        _CACHE[dataset] = cached
        _META[dataset] = {
            "adapter": "spine-store",
            "n_traces": len(cached),
            "truncated": False,
            **_stats(cached),
        }
        while len(_CACHE) > MAX_DATASETS:
            evicted, _ = _CACHE.popitem(last=False)
            _META.pop(evicted, None)
        return cached

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
        "median_cot_tokens": int(median([t.cot_tokens for t in traces])),
        "exact_dup_rate": round(1 - uniq / len(traces), 3),
        "n_models": len({t.agent for t in traces}),
    }


def _action_mix(traces: list[CachedTrace]) -> dict[str, float]:
    """Normalized frequency of each atom across the given traces."""
    counts: Counter[str] = Counter(a for t in traces for a in t.atoms)
    total = sum(counts.values()) or 1
    return {a: round(n / total, 4) for a, n in counts.most_common()}


# API.
app = FastAPI(title="ProcGrep explorer", docs_url="/api")

# Allow the published Pages explorer to call the live query/compare API, so the
# static site can run example queries over whole datasets (not just its embedded
# samples). Read-only JSON endpoints; no credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hamidah.me",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    dataset: str = SUGGESTED[0]
    pattern: str  # regex over the space-joined atom spine
    offset: int = 0  # paginate the matched-trace sample for "show more"


@app.get("/datasets")
def datasets() -> JSONResponse:
    """Suggested datasets plus which ones are already warm in the cache.

    Store-backed datasets are surfaced first (they answer instantly), followed
    by the curated suggestions, deduped in that order.
    """
    store = _load_store()
    store_ids = list(store.keys())
    suggested = list(dict.fromkeys([*store_ids, *SUGGESTED]))
    # datasets that carry a genuine resolution label, so the outcome axis applies.
    outcome_datasets = [
        d for d, ts in store.items() if any(t.outcome in ("resolved", "unresolved") for t in ts)
    ]
    return JSONResponse(
        {
            "suggested": suggested,
            "cached": list(_CACHE.keys()),
            "meta": _META,
            "outcome_datasets": outcome_datasets,
        }
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
                    "median_cot_tokens",
                    "exact_dup_rate",
                    "n_models",
                )
            },
            "by_model": models,
            "mix_all": _action_mix(traces),
            "mix_hits": _action_mix(hits) if hits else {},
            "atom_color": ATOM_COLOR,
            "n_shown_from": req.offset,
            "hits": [
                {
                    "trace_id": t.trace_id,
                    "model": t.agent,
                    "task": t.task,
                    "outcome": t.outcome,
                    "cot_tokens": t.cot_tokens,
                    "steps": len(t.atoms),
                    "atoms": list(t.atoms[:200]),
                }
                for t in hits[req.offset : req.offset + HIT_SAMPLE]
            ],
        }
    )


class CompareRequest(BaseModel):
    """A side-by-side comparison along one axis.

    axis="agent": ``dataset`` is one store dataset; ``left``/``right`` are agent
    names within it. axis="eval": ``left``/``right`` are dataset ids (``dataset``
    is ignored). axis="outcome" is reserved for the outcome-column build.
    """

    axis: str = "agent"
    dataset: str = SUGGESTED[0]
    left: str
    right: str


def _side_traces(axis: str, dataset: str, value: str) -> tuple[list[CachedTrace], str]:
    """Resolve one side of a comparison to its traces and a display label."""
    if axis == "eval":
        # org/short label so same-basename datasets stay distinct, e.g.
        # nebius/SWE-agent vs ElenaFu/SWE-agent.
        org, _, name = value.partition("/")
        short = name.replace("-trajectories", "") or name
        return _load(value), f"{org}/{short}" if org else name
    if axis == "agent":
        return [t for t in _load(dataset) if t.agent == value], value
    if axis == "outcome":
        # value is "resolved" or "unresolved"; only datasets carrying a genuine
        # resolution label have these, so other datasets yield empty sides.
        return [t for t in _load(dataset) if t.outcome == value], value
    raise ValueError(f"unsupported axis: {axis}")


def _mix_vector(
    mix_a: dict[str, float], mix_b: dict[str, float]
) -> tuple[list[float], list[float]]:
    """Align two action-mix dicts onto a shared atom alphabet for JSD."""
    keys = sorted(set(mix_a) | set(mix_b))
    return [mix_a.get(k, 0.0) for k in keys], [mix_b.get(k, 0.0) for k in keys]


def _side_summary(traces: list[CachedTrace], label: str) -> dict:
    """Trail stack plus the same quick stats the query view reports.

    The stack is an even spread across the length distribution (short to long),
    not the longest N, so it reads as a fingerprint of the whole group rather
    than its tail.
    """
    ordered = sorted((t for t in traces if t.atoms), key=lambda t: len(t.atoms))
    if len(ordered) > CMP_STACK:
        stride = len(ordered) / CMP_STACK
        picked = [ordered[int(i * stride)] for i in range(CMP_STACK)]
    else:
        picked = ordered
    return {
        "label": label,
        "n": len(traces),
        "stats": _stats(traces),
        "mix": _action_mix(traces),
        "trails": [
            {
                "trace_id": t.trace_id,
                "model": t.agent,
                "task": t.task,
                "outcome": t.outcome,
                "cot_tokens": t.cot_tokens,
                "steps": len(t.atoms),  # true length; atoms below are capped for rendering
                "atoms": list(t.atoms[:CMP_TRAIL_CAP]),
            }
            for t in picked
        ],
    }


def _compute_compare(axis: str, dataset: str, left_value: str, right_value: str) -> dict:
    """Build the side-by-side diff payload, or an ``{"error": ...}`` dict.

    Trail stacks plus a diff strip: the top procedures distinguishing the two
    sides (signed log-odds, positive favors the left), the action-mix JSD, and
    median length/CoT deltas. Reuses the library's BPE + discriminative_procedures
    (design decision 6).
    """
    try:
        left, left_label = _side_traces(axis, dataset, left_value)
        right, right_label = _side_traces(axis, dataset, right_value)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if left_label == right_label:
        return {"error": "pick two different groups"}
    if not left or not right:
        return {"error": f"no traces for {(left_label if not left else right_label)!r}"}

    t0 = time.perf_counter()
    # Build labeled Traces (group = side) for a sampled BPE + discriminative pass.
    sample = [
        *(Trace(t.trace_id, t.agent, t.atoms, group=left_label) for t in left[:CMP_SAMPLE]),
        *(Trace(t.trace_id, t.agent, t.atoms, group=right_label) for t in right[:CMP_SAMPLE]),
    ]
    procedures: list[dict] = []
    try:
        vocab = fit_bpe((t.atoms for t in sample), vocab_size=CMP_VOCAB)
        fingerprints = encode(sample, vocab=vocab)
        for d in discriminative_procedures(
            fingerprints, vocab, group_a=left_label, group_b=right_label, k=10
        ):
            procedures.append(
                {
                    "atoms": d.procedure.split(PROCEDURE_SEPARATOR),
                    "log_odds": round(d.log_odds, 3),
                    "p_left": round(d.p_a, 4),
                    "p_right": round(d.p_b, 4),
                }
            )
    except Exception:  # diff is best-effort; the stacks still render without it
        procedures = []

    lsum, rsum = _side_summary(left, left_label), _side_summary(right, right_label)
    pl, pr = _mix_vector(lsum["mix"], rsum["mix"])
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return {
        "axis": axis,
        "left": lsum,
        "right": rsum,
        "diff": {
            "procedures": procedures,
            "jsd": round(jsd(pl, pr), 4) if pl else None,
            "len_delta": lsum["stats"].get("median_len", 0) - rsum["stats"].get("median_len", 0),
            "cot_delta": lsum["stats"].get("median_cot", 0) - rsum["stats"].get("median_cot", 0),
        },
        "atom_color": ATOM_COLOR,
        "elapsed_ms": round(elapsed_ms, 2),
    }


@app.post("/compare")
def compare(req: CompareRequest) -> JSONResponse:
    """Cached side-by-side diff. Cache key is (axis, dataset, left, right).

    The default landing pair is pre-warmed at startup (design decision 7) so the
    comparator answers instantly rather than paying the first BPE pass live.
    """
    key = (req.axis, req.dataset, req.left, req.right)
    cached = _COMPARE_CACHE.get(key)
    if cached is not None:
        _COMPARE_CACHE.move_to_end(key)
        return JSONResponse({**cached, "cached": True})
    payload = _compute_compare(req.axis, req.dataset, req.left, req.right)
    if "error" not in payload:
        _COMPARE_CACHE[key] = payload
        while len(_COMPARE_CACHE) > COMPARE_CACHE_MAX:
            _COMPARE_CACHE.popitem(last=False)
    return JSONResponse(payload)


def _warm_default_compare() -> None:
    """Cache the comparator's default landing pair for every axis (agent, eval,
    outcome), mirroring the frontend defaults so each axis lands instantly.
    Runs in a background thread so it never blocks startup.
    """
    try:
        store = _load_store()
        datasets = list(store.keys()) or list(SUGGESTED)
    except Exception:
        return

    def warm(axis: str, dataset: str, left: str, right: str) -> None:
        key = (axis, dataset, left, right)
        if key not in _COMPARE_CACHE:
            with contextlib.suppress(Exception):
                _COMPARE_CACHE[key] = _compute_compare(axis, dataset, left, right)

    # agent: first dataset, its two most common agents (frontend default).
    try:
        agents = [a for a, _ in Counter(t.agent for t in _load(datasets[0])).most_common()]
        if len(agents) >= 2:
            warm("agent", datasets[0], agents[0], agents[1])
    except Exception:
        pass
    # eval: first two store datasets.
    if len(datasets) >= 2:
        warm("eval", datasets[0], datasets[0], datasets[1])
    # outcome: first dataset that carries a resolved label, resolved vs unresolved.
    outcome_ds = [
        d for d, ts in store.items() if any(t.outcome in ("resolved", "unresolved") for t in ts)
    ]
    if outcome_ds:
        warm("outcome", outcome_ds[0], "resolved", "unresolved")


@app.on_event("startup")
def _warm_on_startup() -> None:
    threading.Thread(target=_warm_default_compare, daemon=True).start()


@app.get("/groups")
def groups(dataset: str = SUGGESTED[0]) -> JSONResponse:
    """Agents present in a dataset (for the comparator's agent picker)."""
    try:
        traces = _load(dataset)
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=200)
    counts = Counter(t.agent for t in traces)
    agents = [{"agent": a, "n": n} for a, n in counts.most_common()]
    return JSONResponse({"dataset": dataset, "agents": agents})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
