"""Shared fixtures for the `procgrep` test suite.

Two corpora are reused across the analysis tests:

* ``small_corpus``: a handful of short sequences with an obvious
  repeated bigram, suitable for asserting that BPE picks the correct
  most-frequent pair and that downstream counting is correct.
* ``structured_corpus``: two agents with deliberately disjoint procedure
  preferences across three groups. JSD between groups is high, and a
  leave-one-group-out probe should predict the agent label well.
"""

from __future__ import annotations

import pytest

from procgrep.types import (
    ATOM_EDIT,
    ATOM_LOCALIZE,
    ATOM_READ_FILE,
    ATOM_RUN_TEST,
    ATOM_SEARCH_REPO,
    ATOM_SUBMIT,
    Trace,
)


@pytest.fixture
def small_corpus() -> list[Trace]:
    """A tiny corpus where (edit, run_test) is the most frequent pair."""
    return [
        Trace(
            trace_id="t1",
            agent="alpha",
            atoms=[ATOM_LOCALIZE, ATOM_EDIT, ATOM_RUN_TEST, ATOM_EDIT, ATOM_RUN_TEST],
        ),
        Trace(
            trace_id="t2",
            agent="alpha",
            atoms=[ATOM_READ_FILE, ATOM_EDIT, ATOM_RUN_TEST],
        ),
        Trace(
            trace_id="t3",
            agent="beta",
            atoms=[ATOM_EDIT, ATOM_RUN_TEST, ATOM_SUBMIT],
        ),
    ]


@pytest.fixture
def structured_corpus() -> list[Trace]:
    """Two agents with very different procedure preferences across three groups.

    Agent ``editor`` heavily uses edit + run_test; agent ``searcher``
    heavily uses search + read. Each agent appears in three groups
    (X, Y, Z) with three trajectories per (agent, group) cell.
    """
    traces: list[Trace] = []
    counter = 0
    for group in ("X", "Y", "Z"):
        for _ in range(3):
            counter += 1
            traces.append(
                Trace(
                    trace_id=f"e{counter}",
                    agent="editor",
                    group=group,
                    atoms=[
                        ATOM_LOCALIZE,
                        ATOM_EDIT,
                        ATOM_RUN_TEST,
                        ATOM_EDIT,
                        ATOM_RUN_TEST,
                        ATOM_EDIT,
                        ATOM_SUBMIT,
                    ],
                )
            )
        for _ in range(3):
            counter += 1
            traces.append(
                Trace(
                    trace_id=f"s{counter}",
                    agent="searcher",
                    group=group,
                    atoms=[
                        ATOM_SEARCH_REPO,
                        ATOM_READ_FILE,
                        ATOM_SEARCH_REPO,
                        ATOM_READ_FILE,
                        ATOM_LOCALIZE,
                        ATOM_SUBMIT,
                    ],
                )
            )
    return traces
