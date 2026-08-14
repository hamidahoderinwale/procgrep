"""Intent: measure a harness's same-condition noise floor -- the JSD that two
disjoint groups of identically-configured runs show against each other -- and
report how many runs a target effect size needs. `procgrep floor` wraps
:func:`measure_floor`; read this when changing what the floor measures.

Identically configured agents already differ run to run, so a claimed
intervention effect must be read against this floor, not against zero. The
floor decays roughly as 1/n with group size n, and at small n it spans real
cross-condition differences. All numbers are relative to the one vocabulary
fit for the run, so every block carries its `procgrep.bpe.VocabSpec`.

Design decisions (benefit / price):

1. Compose, never reimplement: distributions come from `fit_bpe` + `encode`,
   divergence from `jsd`, with group means formed exactly as in `jsd_matrix`.
   Benefit: the floor is measured by the same instrument as the effects it
   calibrates. Price: the floor only moves as fast as the library surface.
2. One vocabulary over the whole corpus, never per cell. Benefit: floors and
   cross-condition differences land on the same scale (JSD is
   vocabulary-relative). Price: a small cell's procedures can be diluted by
   the rest of the corpus.
3. Monte Carlo at the measured n, never closed form: each grid point draws
   disjoint n-vs-n splits at that same n with the same estimator. Benefit:
   the percentiles are honest at every n. Price: ``n_draws`` JSD evaluations
   per grid point.
4. Repeated trace_ids within a condition are treated as distinct rollouts,
   never deduplicated -- a repeat is usually a genuine re-run, which is
   exactly what a floor is made of. Benefit: re-runs keep their weight.
   Price: true duplicate rows drag the floor down, so a warning names the
   repeat count for the caller to check.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from procgrep.bpe import VocabSpec, fit_bpe
from procgrep.encode import encode
from procgrep.jsd import jsd
from procgrep.types import Trace

MIN_GROUP_SIZE = 5
"""Smallest group size on the floor curve; a cell needs ``2 * MIN_GROUP_SIZE``
trajectories (two disjoint groups) to be measured at all."""

DEFAULT_DELTAS: tuple[float, ...] = (0.05, 0.1, 0.2)
"""Default target detectable differences for the seeds-needed table."""


@dataclass(frozen=True)
class FloorPoint:
    """The floor at one group size.

    Attributes:
        n: Group size; each draw compares two disjoint groups of ``n`` runs.
        n_draws: Monte Carlo draws behind the estimates.
        mean: Mean JSD between the two group-mean fingerprints across draws.
        p2_5: 2.5th percentile of the draws.
        p97_5: 97.5th percentile of the draws. An observed effect at group
            size ``n`` is only readable above this value.
    """

    n: int
    n_draws: int
    mean: float
    p2_5: float
    p97_5: float

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping of every field."""
        return {
            "n": self.n,
            "n_draws": self.n_draws,
            "mean": self.mean,
            "p2_5": self.p2_5,
            "p97_5": self.p97_5,
        }


@dataclass(frozen=True)
class SeedsNeeded:
    """Runs needed per arm to detect a difference of ``delta``.

    Attributes:
        delta: Target detectable JSD difference between two arms.
        n: Smallest grid group size whose floor 97.5th percentile is below
            ``delta``; ``None`` when even the largest supported group size
            does not get the floor under ``delta``.
    """

    delta: float
    n: int | None

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping of every field."""
        return {"delta": self.delta, "n": self.n}


@dataclass(frozen=True)
class CellFloor:
    """The floor curve for one condition cell.

    Attributes:
        condition: The cell's value of the condition field.
        n_traces: Non-empty trajectories in the cell.
        n_repeated_trace_ids: Extra occurrences of already-seen trace_ids;
            repeats are kept as distinct rollouts, never deduplicated.
        curve: One `FloorPoint` per grid group size, ascending; the last
            point is always the full-n floor.
        full_n: Largest group size the cell supports (``n_traces // 2``).
        seeds_needed: One `SeedsNeeded` row per requested delta.
        vocab_spec: Compact key (``content_hash:vocab_size``) of the shared
            vocabulary; see `procgrep.bpe.VocabSpec`.
    """

    condition: str
    n_traces: int
    n_repeated_trace_ids: int
    curve: tuple[FloorPoint, ...]
    full_n: int
    seeds_needed: tuple[SeedsNeeded, ...]
    vocab_spec: str

    @property
    def full_n_floor(self) -> FloorPoint:
        """The floor at the largest group size the cell supports."""
        return self.curve[-1]

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping; ``floor`` holds the curve points."""
        return {
            "condition": self.condition,
            "n_traces": self.n_traces,
            "n_repeated_trace_ids": self.n_repeated_trace_ids,
            "vocab_spec": self.vocab_spec,
            "floor": [point.to_dict() for point in self.curve],
            "full_n": self.full_n,
            "full_n_floor": self.full_n_floor.to_dict(),
            "seeds_needed": [row.to_dict() for row in self.seeds_needed],
        }


@dataclass(frozen=True)
class RefusedCell:
    """A condition cell the floor refused to measure, and why."""

    condition: str
    n_traces: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping of every field."""
        return {"condition": self.condition, "n_traces": self.n_traces, "reason": self.reason}


@dataclass(frozen=True)
class FloorReport:
    """Noise-floor measurement for one corpus.

    Attributes:
        source: Label echoed from the caller (dataset id or file name).
        condition: The field whose values identify identically-configured runs.
        n_traces: Trajectories given, including empty ones.
        n_empty: Trajectories with zero atoms, excluded from every cell.
        n_missing_condition: Non-empty trajectories with no value for the
            condition field, excluded from every cell.
        seed: Seed behind both the vocabulary fit and the Monte Carlo draws.
        n_draws: Monte Carlo draws per grid point.
        deltas: Target detectable differences of the seeds-needed table.
        vocab_spec: Full spec of the one vocabulary fit over the whole
            corpus; every cell block carries its compact key.
        cells: One `CellFloor` per measurable condition cell, sorted by name.
        refused: Cells with too few trajectories to measure.
    """

    source: str
    condition: str
    n_traces: int
    n_empty: int
    n_missing_condition: int
    seed: int
    n_draws: int
    deltas: tuple[float, ...]
    vocab_spec: VocabSpec
    cells: tuple[CellFloor, ...]
    refused: tuple[RefusedCell, ...]

    def to_dict(self) -> dict[str, object]:
        """JSON-ready mapping of every field."""
        return {
            "source": self.source,
            "condition": self.condition,
            "n_traces": self.n_traces,
            "n_empty": self.n_empty,
            "n_missing_condition": self.n_missing_condition,
            "seed": self.seed,
            "n_draws": self.n_draws,
            "deltas": list(self.deltas),
            "vocab_spec": self.vocab_spec.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "refused": [cell.to_dict() for cell in self.refused],
        }


def measure_floor(
    traces: Sequence[Trace],
    *,
    condition: str,
    source: str = "traces",
    vocab_size: int = 64,
    seed: int = 0,
    n_draws: int = 200,
    deltas: Sequence[float] = DEFAULT_DELTAS,
) -> FloorReport:
    """Measure the same-condition noise floor of a trajectory corpus.

    Groups ``traces`` into cells by their value of ``condition`` -- the field
    that identifies identically-configured runs (``"agent"``, ``"group"``, or
    any metadata key). One BPE vocabulary is fit over the whole corpus; then,
    for each cell and each group size n on a 5 / 10 / 25 / 50 / ... grid up
    to half the cell, ``n_draws`` Monte Carlo draws split 2n runs into two
    disjoint groups of n and record the JSD between the group-mean
    fingerprints. That curve is the divergence the harness produces from
    run-to-run variation alone; an intervention effect measured at group
    size n is only readable above the curve's 97.5th percentile at that n.

    Deterministic: the same traces, condition, and arguments reproduce the
    report bit for bit (draws are seeded per cell and group size).

    Args:
        condition: ``"agent"``, ``"group"``, or a metadata key naming the
            column that identifies identically-configured runs.
        source: Label echoed in the report.
        vocab_size: BPE procedure vocabulary size.
        seed: Seeds the vocabulary fit and the Monte Carlo draws.
        n_draws: Monte Carlo draws per grid point.
        deltas: Target detectable differences for the seeds-needed table.

    Warns:
        UserWarning: A cell repeats trace_ids. Repeats are treated as
            distinct rollouts by design (a re-run under the same id is
            still a rollout) and are never deduplicated; check for true
            duplicate rows if the warning is unexpected.

    Raises:
        ValueError: Empty corpus, no trace carries the condition field, or
            no cell has the ``2 * MIN_GROUP_SIZE`` trajectories the smallest
            group size needs (per-cell shortfalls land in ``refused``
            instead when at least one cell is measurable).
    """
    if not traces:
        raise ValueError("cannot measure a floor on an empty corpus")

    parsed = [t for t in traces if t.atoms]
    n_empty = len(traces) - len(parsed)

    by_cell: dict[str, list[Trace]] = {}
    n_missing = 0
    for trace in parsed:
        value = _condition_value(trace, condition)
        if value is None:
            n_missing += 1
        else:
            by_cell.setdefault(value, []).append(trace)
    if not by_cell:
        raise ValueError(
            f"condition field {condition!r} is missing on every trace; pass 'agent', "
            "'group', or a metadata key present in the corpus"
        )

    vocab = fit_bpe((t.atoms for t in parsed), vocab_size=vocab_size, seed=seed, fit_corpus=source)
    spec = vocab.spec
    delta_grid = tuple(deltas)

    cells: list[CellFloor] = []
    refused: list[RefusedCell] = []
    for name in sorted(by_cell):
        members = by_cell[name]
        if len(members) < 2 * MIN_GROUP_SIZE:
            refused.append(
                RefusedCell(
                    condition=name,
                    n_traces=len(members),
                    reason=(
                        f"{len(members)} trajectories < {2 * MIN_GROUP_SIZE} needed: "
                        f"the smallest group size n={MIN_GROUP_SIZE} requires two "
                        "disjoint groups of n"
                    ),
                )
            )
            continue

        n_repeats = _repeated_trace_ids(members)
        if n_repeats:
            warnings.warn(
                f"condition {name!r} repeats {n_repeats} trace_id value(s); repeats are "
                "treated as distinct rollouts and never deduplicated -- if they are "
                "duplicate rows of one rollout, the floor reads too low",
                stacklevel=2,
            )

        distributions = np.stack([fp.distribution() for fp in encode(members, vocab=vocab)])
        full_n = len(members) // 2
        curve = tuple(
            _floor_point(
                distributions,
                n=n,
                n_draws=n_draws,
                rng=np.random.default_rng([seed, _cell_key(name), n]),
            )
            for n in _n_grid(full_n)
        )
        cells.append(
            CellFloor(
                condition=name,
                n_traces=len(members),
                n_repeated_trace_ids=n_repeats,
                curve=curve,
                full_n=full_n,
                seeds_needed=_seeds_needed(curve, delta_grid),
                vocab_spec=spec.compact(),
            )
        )

    if not cells:
        shortfalls = ", ".join(f"{r.condition}={r.n_traces}" for r in refused)
        raise ValueError(
            f"no condition cell has the {2 * MIN_GROUP_SIZE} trajectories the smallest "
            f"group size n={MIN_GROUP_SIZE} needs (cells: {shortfalls}); collect more "
            "runs per condition or check that the condition field is the right column"
        )

    return FloorReport(
        source=source,
        condition=condition,
        n_traces=len(traces),
        n_empty=n_empty,
        n_missing_condition=n_missing,
        seed=seed,
        n_draws=n_draws,
        deltas=delta_grid,
        vocab_spec=spec,
        cells=tuple(cells),
        refused=tuple(refused),
    )


def _condition_value(trace: Trace, condition: str) -> str | None:
    """The trace's value of the condition field, or None when absent."""
    if condition == "agent":
        return trace.agent
    if condition == "group":
        return trace.grouping()
    value = trace.metadata.get(condition)
    return None if value is None else str(value)


def _repeated_trace_ids(traces: Sequence[Trace]) -> int:
    """Occurrences of trace_ids beyond each id's first."""
    counts = Counter(t.trace_id for t in traces)
    return sum(c - 1 for c in counts.values())


def _cell_key(condition: str) -> int:
    """Stable per-cell RNG stream key, independent of cell order."""
    digest = hashlib.sha256(condition.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _n_grid(max_n: int) -> list[int]:
    """Group sizes 5, 10, 25, 50, 100, ... capped at (and ending with) max_n."""
    grid: list[int] = []
    scale = 1
    while True:
        for base in (5, 10, 25):
            n = base * scale
            if n >= max_n:
                grid.append(max_n)
                return grid
            grid.append(n)
        scale *= 10


def _floor_point(
    distributions: npt.NDArray[np.float64],
    *,
    n: int,
    n_draws: int,
    rng: np.random.Generator,
) -> FloorPoint:
    """The floor at one group size, by Monte Carlo n-vs-n splits.

    Each draw samples ``2n`` distinct rows, splits them into two disjoint
    groups of ``n``, and takes the JSD between the group means -- the same
    mean-then-normalize aggregation `jsd_matrix` uses, at the sample size
    being measured (a closed-form or full-sample floor would sit below the
    values small-n measurements actually produce).
    """
    n_rows = distributions.shape[0]
    draws = np.empty(n_draws, dtype=np.float64)
    for i in range(n_draws):
        picked = rng.choice(n_rows, size=2 * n, replace=False)
        mean_a = distributions[picked[:n]].mean(axis=0)
        mean_b = distributions[picked[n:]].mean(axis=0)
        draws[i] = jsd(mean_a, mean_b)
    p_lo, p_hi = np.percentile(draws, [2.5, 97.5])
    return FloorPoint(
        n=n,
        n_draws=n_draws,
        mean=float(draws.mean()),
        p2_5=float(p_lo),
        p97_5=float(p_hi),
    )


def _seeds_needed(
    curve: Sequence[FloorPoint],
    deltas: Sequence[float],
) -> tuple[SeedsNeeded, ...]:
    """Smallest grid n whose floor 97.5th percentile is below each delta."""
    out: list[SeedsNeeded] = []
    for delta in deltas:
        n = next((point.n for point in curve if point.p97_5 < delta), None)
        out.append(SeedsNeeded(delta=delta, n=n))
    return tuple(out)


__all__ = [
    "DEFAULT_DELTAS",
    "MIN_GROUP_SIZE",
    "CellFloor",
    "FloorPoint",
    "FloorReport",
    "RefusedCell",
    "SeedsNeeded",
    "measure_floor",
]
