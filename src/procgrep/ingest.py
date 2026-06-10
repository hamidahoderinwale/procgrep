"""Dynamic, schema-sniffing ingestion of trajectory datasets from the Hub.

Given only a dataset id, infer (a) which adapter produced the traces and
(b) which columns hold the trace id, agent, and group -- by inspecting the
dataset's schema and a few sample rows, never by a hard-coded dataset->adapter
table. Adding support for a new trace format means adding a :class:`Sniffer`,
not a dataset mapping.

Pipeline (each step is cheap until the last):

    introspect(dataset)  -> DatasetSchema   (HF datasets-server, no download)
    sniff(schema)        -> ranked SniffResults
    plan(dataset)        -> IngestionPlan    (adapter + field map + sample atoms)
    ingest(dataset)      -> list[Trace]      (stream + limit; via procgrep.hf)

Introspection uses the public datasets-server REST API
(``/splits`` + ``/first-rows``), which is parquet-backed and returns column
features plus a handful of sample rows without materializing the dataset.
``--dry-run`` (see :func:`plan`) surfaces the inferred plan and a sample of
canonical atoms so a human can verify the auto-detection before a large run.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from procgrep.canonicalize import canonicalize, get_adapter
from procgrep.types import Trace

_DATASETS_SERVER = "https://datasets-server.huggingface.co"

# Candidate column names, most-specific first, for the three identity fields.
_ID_FIELDS = ("trace_id", "instance_id", "traj_id", "task_id", "id", "run_id")
_AGENT_FIELDS = ("agent", "model", "model_name", "generator", "run_id")
_GROUP_FIELDS = ("repo", "repository", "group", "language")

# Columns that may hold the conversation/turn list, most-specific first. A cell
# may be a Python list OR a JSON-encoded string (datasets-server serializes
# nested columns, and some datasets store the whole trajectory as one string).
_CONV_COLUMNS = ("messages", "trajectory", "traj", "history", "conversations", "steps")


def _decode_conv(value: Any) -> list[Any] | None:
    """Coerce a conversation cell to a list of turns, or None.

    Handles the two real encodings: a native list, or a JSON string. Returns
    None on anything else or on a truncated/invalid JSON string (the sample may
    be truncated by the datasets-server; the full row decodes at ingest time).
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def _locate_turns(row: Mapping[str, Any]) -> tuple[str | None, list[Any] | None]:
    """Find and decode the first present conversation column."""
    for col in _CONV_COLUMNS:
        if col in row:
            turns = _decode_conv(row[col])
            if turns is not None:
                return col, turns
    return None, None


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the decoded turn list under ``messages`` so adapters find it."""
    col, turns = _locate_turns(row)
    out = dict(row)
    if turns is not None and col != "messages":
        out["messages"] = turns
    elif turns is not None:
        out["messages"] = turns  # decode an in-place JSON-string `messages`
    return out


@dataclass(frozen=True)
class DatasetSchema:
    """Cheap, download-free view of a Hub dataset."""

    dataset: str
    config: str
    split: str
    columns: tuple[str, ...]
    sample_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class Sniffer:
    """A trace-format detector bound to a registered adapter.

    ``detect`` returns a confidence in [0, 1] that ``schema`` was produced by
    this adapter's scaffold, judged from columns and sample rows.
    """

    adapter: str
    detect: Callable[[DatasetSchema], float]


@dataclass(frozen=True)
class SniffResult:
    adapter: str
    confidence: float


@dataclass(frozen=True)
class IngestionPlan:
    """Everything needed to ingest a dataset, inferred not hard-coded."""

    dataset: str
    config: str
    split: str
    adapter: str
    confidence: float
    trace_id_field: str
    agent_field: str
    group_field: str | None
    sample_atoms: tuple[tuple[str, ...], ...] = ()
    candidates: tuple[SniffResult, ...] = ()
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [
            f"dataset    {self.dataset}  [{self.config}/{self.split}]",
            f"adapter    {self.adapter}  (confidence {self.confidence:.2f})",
            f"fields     id={self.trace_id_field}  agent={self.agent_field}  "
            f"group={self.group_field}",
        ]
        if self.candidates:
            ranked = ", ".join(f"{c.adapter}={c.confidence:.2f}" for c in self.candidates)
            lines.append(f"considered {ranked}")
        for atoms in self.sample_atoms[:3]:
            preview = " ".join(atoms[:24]) + (" …" if len(atoms) > 24 else "")
            lines.append(f"  sample atoms  {preview or '(empty)'}")
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


# ── introspection ────────────────────────────────────────────────────────────


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "procgrep-ingest"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode())
    return data


def introspect(
    dataset: str,
    *,
    config: str | None = None,
    split: str | None = None,
    timeout: float = 30.0,
) -> DatasetSchema:
    """Fetch columns + sample rows via the datasets-server (no download).

    Picks the first config/split when unspecified.
    """
    splits_url = f"{_DATASETS_SERVER}/splits?dataset={urllib.parse.quote(dataset)}"
    splits = _get_json(splits_url, timeout=timeout).get("splits", [])
    if not splits:
        raise ValueError(f"no splits reported for dataset {dataset!r}")
    chosen = next(
        (
            s
            for s in splits
            if (config is None or s["config"] == config) and (split is None or s["split"] == split)
        ),
        splits[0],
    )
    cfg, spl = chosen["config"], chosen["split"]
    rows_url = (
        f"{_DATASETS_SERVER}/first-rows?dataset={urllib.parse.quote(dataset)}"
        f"&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(spl)}"
    )
    payload = _get_json(rows_url, timeout=timeout)
    columns = tuple(f["name"] for f in payload.get("features", []))
    sample_rows = tuple(r["row"] for r in payload.get("rows", []))
    return DatasetSchema(dataset, cfg, spl, columns, sample_rows)


# ── sniffers ─────────────────────────────────────────────────────────────────


def _assistant_messages(schema: DatasetSchema) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for row in schema.sample_rows:
        msgs = _normalize_row(row).get("messages")
        if isinstance(msgs, list):
            out.extend(m for m in msgs if isinstance(m, Mapping) and m.get("role") == "assistant")
    return out


def _sniff_openhands(schema: DatasetSchema) -> float:
    # Don't gate on a literal `messages` column: the turns may live under
    # `trajectory` etc. Let the locator find them, then look for tool_calls.
    asst = _assistant_messages(schema)
    if any(isinstance(m.get("tool_calls"), list) and m["tool_calls"] for m in asst):
        return 0.95
    return 0.4 if "tools" in schema.columns else 0.0


def _sniff_mini_swe(schema: DatasetSchema) -> float:
    for m in _assistant_messages(schema):
        extra = m.get("extra")
        if isinstance(extra, Mapping) and isinstance(extra.get("actions"), list):
            return 0.95
    return 0.0


def _sniff_swe_agent(schema: DatasetSchema) -> float:
    # SWE-agent .traj turns are {action, observation, ...} with no `role`
    # (distinguishing them from role-based message turns).
    for row in schema.sample_rows:
        _, turns = _locate_turns(row)
        if turns and isinstance(turns[0], Mapping):
            t0 = turns[0]
            if "action" in t0 and "observation" in t0 and "role" not in t0:
                return 0.9
    # A trajectory-shaped column whose sample was truncated: tentative.
    if any(c in schema.columns for c in ("trajectory", "traj", "history")):
        return 0.45
    return 0.0


def _sniff_swe_smith(schema: DatasetSchema) -> float:
    # SWE-agent ReAct serialized as messages (text in `content`, no tool_calls),
    # tagged with instance_id + model.
    if "messages" not in schema.columns:
        return 0.0
    has_ids = "instance_id" in schema.columns and "model" in schema.columns
    asst = _assistant_messages(schema)
    no_tool_calls = asst and not any(m.get("tool_calls") for m in asst)
    return 0.7 if (has_ids and no_tool_calls) else 0.0


SNIFFERS: tuple[Sniffer, ...] = (
    Sniffer("openhands", _sniff_openhands),
    Sniffer("mini-swe-agent", _sniff_mini_swe),
    Sniffer("swe-agent", _sniff_swe_agent),
    Sniffer("swe-smith", _sniff_swe_smith),
)


def sniff(schema: DatasetSchema) -> list[SniffResult]:
    """Rank registered sniffers by confidence (descending, ties broken by order)."""
    scored = [SniffResult(s.adapter, max(0.0, min(1.0, s.detect(schema)))) for s in SNIFFERS]
    return sorted(scored, key=lambda r: r.confidence, reverse=True)


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((c for c in candidates if c in columns), None)


def _probe_full_rows(dataset: str, config: str, split: str, *, n: int = 3) -> list[dict[str, Any]]:
    """Stream a few *untruncated* rows via ``datasets`` for accurate sniffing.

    The datasets-server ``/first-rows`` truncates large nested cells, so a
    trajectory stored as one big JSON string is unreadable from the sample.
    Pulling a handful of full rows (streaming, no full download) fixes sniffing
    for those datasets. Returns ``[]`` if ``datasets`` is unavailable.
    """
    try:
        from procgrep.hf import _import_load_dataset

        load_dataset = _import_load_dataset()
        ds = load_dataset(dataset, name=config, split=split, streaming=True)
        return [dict(r) for r in ds.take(n)]
    except Exception:
        return []


# ── plan + ingest ────────────────────────────────────────────────────────────


def plan(
    dataset: str,
    *,
    config: str | None = None,
    split: str | None = None,
    adapter: str | None = None,
    trace_id_field: str | None = None,
    agent_field: str | None = None,
    group_field: str | None = None,
    timeout: float = 30.0,
    probe: bool = True,
) -> IngestionPlan:
    """Infer an :class:`IngestionPlan`; explicit args override inference.

    When the cheap first-rows sample yields a low-confidence guess (e.g. a
    truncated trajectory cell), ``probe`` streams a few full rows to re-sniff.
    """
    schema = introspect(dataset, config=config, split=split, timeout=timeout)
    ranked = sniff(schema)
    notes: list[str] = []

    if probe and (not ranked or ranked[0].confidence < 0.6):
        full_rows = _probe_full_rows(dataset, schema.config, schema.split)
        if full_rows:
            schema = DatasetSchema(
                dataset, schema.config, schema.split, schema.columns, tuple(full_rows)
            )
            ranked = sniff(schema)
            notes.append("re-sniffed on full rows (first-rows sample was truncated)")

    chosen_adapter = adapter or (ranked[0].adapter if ranked and ranked[0].confidence > 0 else None)
    confidence = next((r.confidence for r in ranked if r.adapter == chosen_adapter), 0.0)
    if chosen_adapter is None:
        raise ValueError(
            f"could not infer an adapter for {dataset!r}; columns={schema.columns}. "
            f"Pass adapter= explicitly or add a Sniffer."
        )
    if confidence < 0.5:
        notes.append(f"low adapter confidence ({confidence:.2f}); verify with --dry-run")

    id_field = trace_id_field or _first_present(schema.columns, _ID_FIELDS)
    ag_field = agent_field or _first_present(schema.columns, _AGENT_FIELDS)
    gp_field = group_field or _first_present(schema.columns, _GROUP_FIELDS)
    if id_field is None:
        notes.append("no id column matched; falling back to row index at ingest time")
    if ag_field is None:
        notes.append("no agent column matched; agent will default to the dataset name")

    sample_atoms = _sample_atoms(schema, chosen_adapter)
    return IngestionPlan(
        dataset=dataset,
        config=schema.config,
        split=schema.split,
        adapter=chosen_adapter,
        confidence=confidence,
        trace_id_field=id_field or "trace_id",
        agent_field=ag_field or "agent",
        group_field=gp_field,
        sample_atoms=sample_atoms,
        candidates=tuple(ranked),
        notes=tuple(notes),
    )


def _sample_atoms(schema: DatasetSchema, adapter: str) -> tuple[tuple[str, ...], ...]:
    """Canonicalize a couple of sample rows so the plan is verifiable."""
    try:
        fn = get_adapter(adapter)
    except KeyError:
        return ()
    out: list[tuple[str, ...]] = []
    for row in schema.sample_rows[:3]:
        try:
            out.append(tuple(fn(_normalize_row(row))))
        except Exception:
            out.append(())
    return tuple(out)


def ingest(
    dataset: str,
    *,
    limit: int | None = None,
    config: str | None = None,
    split: str | None = None,
    adapter: str | None = None,
    trace_id_field: str | None = None,
    agent_field: str | None = None,
    group_field: str | None = None,
    revision: str | None = None,
    timeout: float = 30.0,
) -> tuple[list[Trace], IngestionPlan]:
    """Plan, then stream + canonicalize ``limit`` rows into Traces.

    Returns the traces and the plan that produced them (so callers can log the
    inferred adapter/fields). Streams via the ``datasets`` library; pair with
    ``limit`` to bound memory and time.
    """
    p = plan(
        dataset,
        config=config,
        split=split,
        adapter=adapter,
        trace_id_field=trace_id_field,
        agent_field=agent_field,
        group_field=group_field,
        timeout=timeout,
    )
    from procgrep.hf import _import_load_dataset

    load_dataset = _import_load_dataset()
    ds = load_dataset(dataset, name=p.config, split=p.split, streaming=True, revision=revision)

    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(ds):
        if limit is not None and i >= limit:
            break
        row = _normalize_row(dict(raw))
        # Graceful field defaults so canonicalize never KeyErrors on a dataset
        # that lacks the inferred id/agent column.
        row.setdefault(p.trace_id_field, str(i))
        row.setdefault(p.agent_field, dataset)
        rows.append(row)

    traces = canonicalize(
        rows,
        adapter=p.adapter,
        trace_id_field=p.trace_id_field,
        agent_field=p.agent_field,
        group_field=p.group_field,
    )
    return traces, p


__all__ = [
    "SNIFFERS",
    "DatasetSchema",
    "IngestionPlan",
    "SniffResult",
    "Sniffer",
    "ingest",
    "introspect",
    "plan",
    "sniff",
]
