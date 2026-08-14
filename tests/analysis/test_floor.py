"""Tests for the same-condition noise floor."""

from __future__ import annotations

import json

import numpy as np
import pytest

from procgrep.floor import MIN_GROUP_SIZE, measure_floor
from procgrep.jsd import jsd
from procgrep.types import Trace


def _synthetic_two_conditions(n_per_condition: int = 60, length: int = 30) -> list[Trace]:
    """Two conditions with within-condition variation and a known separation.

    Each condition draws atoms from its own fixed mixture (edit-heavy vs
    search-heavy), so trajectories vary run to run within a condition while
    the condition means stay far apart.
    """
    rng = np.random.default_rng(7)
    alphabet = ["edit", "run_test", "search_repo", "read_file"]
    mixtures = {
        "edit-heavy": [0.6, 0.25, 0.1, 0.05],
        "search-heavy": [0.05, 0.1, 0.25, 0.6],
    }
    traces: list[Trace] = []
    for cond, weights in mixtures.items():
        for i in range(n_per_condition):
            traces.append(
                Trace(
                    trace_id=f"{cond}-{i}",
                    agent="model-x",
                    atoms=list(rng.choice(alphabet, size=length, p=weights)),
                    metadata={"condition": cond},
                )
            )
    return traces


def test_floor_decays_with_n() -> None:
    report = measure_floor(
        _synthetic_two_conditions(), condition="condition", n_draws=100, source="unit"
    )
    assert len(report.cells) == 2
    for cell in report.cells:
        # grid runs 5, 10, 25 and ends at the full supported group size
        assert [p.n for p in cell.curve] == [5, 10, 25, 30]
        assert cell.full_n == 30
        assert cell.full_n_floor == cell.curve[-1]
        # the same-condition floor decays as group size grows
        assert cell.curve[0].mean > cell.curve[-1].mean
        assert cell.curve[0].p97_5 > cell.curve[-1].p97_5
        for point in cell.curve:
            assert 0.0 <= point.p2_5 <= point.mean <= point.p97_5 <= 1.0


def test_known_separation_clears_the_large_n_floor() -> None:
    traces = _synthetic_two_conditions()
    report = measure_floor(traces, condition="condition", n_draws=100)

    # the cross-condition difference, measured at the atom level, is real and
    # large; the full-n same-condition floor must sit clearly below it
    def mean_dist(cond: str) -> np.ndarray:
        alphabet = ["edit", "run_test", "search_repo", "read_file"]
        counts = np.zeros(len(alphabet))
        for t in traces:
            if t.metadata["condition"] == cond:
                for a in t.atoms:
                    counts[alphabet.index(a)] += 1
        return counts / counts.sum()

    separation = jsd(mean_dist("edit-heavy"), mean_dist("search-heavy"))
    for cell in report.cells:
        assert cell.full_n_floor.p97_5 < separation


def test_deterministic_given_a_seed() -> None:
    traces = _synthetic_two_conditions()
    a = measure_floor(traces, condition="condition", n_draws=50, seed=3)
    b = measure_floor(traces, condition="condition", n_draws=50, seed=3)
    assert json.dumps(a.to_dict()) == json.dumps(b.to_dict())
    # a different seed draws different splits
    c = measure_floor(traces, condition="condition", n_draws=50, seed=4)
    assert json.dumps(a.to_dict()) != json.dumps(c.to_dict())


def test_vocab_spec_stamped_on_every_block() -> None:
    report = measure_floor(_synthetic_two_conditions(), condition="condition", n_draws=25)
    payload = report.to_dict()
    spec = payload["vocab_spec"]
    assert isinstance(spec, dict)
    compact = f"{spec['content_hash']}:{spec['vocab_size']}"
    assert payload["cells"]
    for cell in payload["cells"]:  # type: ignore[union-attr]
        assert cell["vocab_spec"] == compact


def test_seeds_needed_brackets_the_grid() -> None:
    report = measure_floor(
        _synthetic_two_conditions(),
        condition="condition",
        n_draws=50,
        deltas=(1e-9, 0.9),
    )
    for cell in report.cells:
        by_delta = {row.delta: row.n for row in cell.seeds_needed}
        # no group size gets the floor under an impossibly small delta
        assert by_delta[1e-9] is None
        # every floor is below 0.9, so the smallest grid size suffices
        assert by_delta[0.9] == MIN_GROUP_SIZE


def test_empty_corpus_refused() -> None:
    with pytest.raises(ValueError, match="empty corpus"):
        measure_floor([], condition="condition")


def test_condition_missing_everywhere_refused() -> None:
    traces = [Trace(trace_id="t", agent="a", atoms=["edit"] * 8)]
    with pytest.raises(ValueError, match="missing on every trace"):
        measure_floor(traces, condition="nope")


def test_all_cells_too_small_refused() -> None:
    traces = _synthetic_two_conditions(n_per_condition=2 * MIN_GROUP_SIZE - 1)
    with pytest.raises(ValueError, match="no condition cell"):
        measure_floor(traces, condition="condition")


def test_small_cell_refused_while_the_rest_measure() -> None:
    traces = _synthetic_two_conditions()
    traces += [
        Trace(trace_id=f"tiny-{i}", agent="a", atoms=["edit"] * 8, metadata={"condition": "tiny"})
        for i in range(3)
    ]
    report = measure_floor(traces, condition="condition", n_draws=25)
    assert {c.condition for c in report.cells} == {"edit-heavy", "search-heavy"}
    assert len(report.refused) == 1
    assert report.refused[0].condition == "tiny"
    assert report.refused[0].n_traces == 3
    assert "disjoint groups" in report.refused[0].reason


def test_repeated_trace_ids_warn_and_are_kept() -> None:
    traces = _synthetic_two_conditions(n_per_condition=12)
    repeats = [
        Trace(
            trace_id="edit-heavy-0",  # same id as an existing rollout
            agent="model-x",
            atoms=["edit"] * 10,
            metadata={"condition": "edit-heavy"},
        )
        for _ in range(2)
    ]
    with pytest.warns(UserWarning, match="distinct rollouts"):
        report = measure_floor(traces + repeats, condition="condition", n_draws=25)
    cell = next(c for c in report.cells if c.condition == "edit-heavy")
    # never deduplicated: all 14 rollouts stay in the cell
    assert cell.n_traces == 14
    assert cell.n_repeated_trace_ids == 2


def test_condition_can_be_agent_or_group() -> None:
    traces = [
        Trace(
            trace_id=f"t{i}",
            agent=f"agent-{i % 2}",
            group=f"g{i % 2}",
            atoms=["edit", "run_test"] * (2 + i % 3),
        )
        for i in range(24)
    ]
    by_agent = measure_floor(traces, condition="agent", n_draws=10)
    assert {c.condition for c in by_agent.cells} == {"agent-0", "agent-1"}
    by_group = measure_floor(traces, condition="group", n_draws=10)
    assert {c.condition for c in by_group.cells} == {"g0", "g1"}
