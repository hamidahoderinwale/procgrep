"""Conduct tree: distance, tree, poset, next-action forecast on controlled corpora."""

from __future__ import annotations

import random

import numpy as np
import pytest
from scipy.stats import binomtest, norm

from procgrep.conduct_tree import (
    build_conduct_tree,
    conditional_distance,
    distance_matrix,
    encode_cells,
    next_action_scores,
    poset_distance,
    precedence_poset,
    split_half_floor,
    unshared_mass,
)


def _runs(rng: random.Random, n: int, *, test_after_edit: float) -> list[list[str]]:
    """Synthetic agent: search, read, then edit loops that test with a given probability."""
    out = []
    for _ in range(n):
        run = ["search_repo", "read_file", "think"]
        for _ in range(rng.randint(2, 6)):
            run.append("edit")
            if rng.random() < test_after_edit:
                run.append("run_test")
        run.append("submit")
        out.append(run)
    return out


@pytest.fixture
def cells() -> dict[str, list[list[str]]]:
    rng = random.Random(0)
    return {
        "tester_a": _runs(rng, 300, test_after_edit=0.9),
        "tester_b": _runs(rng, 300, test_after_edit=0.9),
        "skipper": _runs(rng, 300, test_after_edit=0.1),
    }


def test_encode_drops_think_and_shares_alphabet(cells):
    encoded = encode_cells(cells)
    assert "think" not in encoded[0].actions
    assert all(c.actions == encoded[0].actions and c.states == encoded[0].states for c in encoded)
    assert encoded[0].counts.shape[0] == 300


def test_identical_policies_sit_at_the_floor_and_different_ones_do_not(cells):
    encoded = encode_cells(cells)
    same = conditional_distance(encoded[0].table(), encoded[1].table())[0]
    diff, contrib = conditional_distance(encoded[0].table(), encoded[2].table())
    floor = split_half_floor(encoded[0], reps=20)
    assert same < 2 * floor["p95"]
    assert diff > 10 * floor["p95"]
    # the split is after `edit`: that state must carry the distance
    assert encoded[0].states[int(np.argmax(contrib))] == "edit"


def test_tree_merges_the_twins_first_with_full_support(cells):
    tree = build_conduct_tree(encode_cells(cells), n_bootstrap=30, floor_reps=10)
    first = tree.merges[0]
    assert set(first.left) | set(first.right) == {"tester_a", "tester_b"}
    assert first.support == 1.0
    last = tree.merges[-1]
    assert last.top_states[0]["state"] == "edit"
    sides = {last.top_states[0]["left_next"]["action"], last.top_states[0]["right_next"]["action"]}
    assert sides == {"run_test", "edit"} or sides == {"run_test", "submit"}


def test_poset_recovers_the_phase_order(cells):
    poset = precedence_poset(cells["tester_a"])
    assert ("search_repo", "read_file") in {tuple(e) for e in poset["hasse"]}
    assert ("read_file", "edit") in {tuple(e) for e in poset["hasse"]}
    # transitive reduction: search -> edit is implied, not drawn
    assert ("search_repo", "edit") not in {tuple(e) for e in poset["hasse"]}
    assert poset_distance(poset, precedence_poset(cells["tester_b"])) < 0.05


def test_own_table_forecasts_better_than_unigram(cells):
    encoded = encode_cells(cells)
    pooled = sum(c.table() for c in encoded)
    scores = next_action_scores(encoded[2], pooled)
    assert scores["own"]["cross_entropy_bits"] < scores["unigram"]["cross_entropy_bits"]
    assert scores["own"]["cross_entropy_bits"] < scores["pooled"]["cross_entropy_bits"]


@pytest.mark.parametrize("run", [["edit", "run_test"], ["run_test", "edit"]])
def test_single_observation_does_not_establish_precedence(run):
    poset = precedence_poset([run])
    interval = poset["pairs"]["edit<run_test"]
    assert interval["n"] == 1
    assert 0 <= interval["lo"] < 0.5 < interval["hi"] <= 1
    assert poset["hasse"] == []


def test_precedence_rejects_cycle_across_runs():
    runs = [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]] * 100
    with pytest.raises(ValueError, match="cycle"):
        precedence_poset(runs)


@pytest.mark.parametrize("empty_side", ["left", "right", "both"])
def test_conditional_distance_rejects_missing_observations(empty_side):
    observed = np.array([[1.0, 0.0], [0.0, 1.0]])
    empty = np.zeros_like(observed)
    left = observed if empty_side == "right" else empty
    right = observed if empty_side == "left" else empty
    with pytest.raises(ValueError, match="observations"):
        conditional_distance(left, right)


def test_conditional_distance_rejects_no_shared_states():
    with pytest.raises(ValueError, match="shared state"):
        conditional_distance(np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]))


def test_empty_condition_does_not_produce_a_tree():
    encoded = encode_cells({"empty": [["think"], ["think"]], "observed": [["edit"], ["edit"]]})
    with pytest.raises(ValueError, match="observations"):
        build_conduct_tree(encoded, n_bootstrap=0, floor_reps=2)


def test_floor_rejects_a_single_run():
    cell = encode_cells({"single": [["edit", "run_test"]]})[0]
    with pytest.raises(ValueError, match="at least two"):
        split_half_floor(cell, reps=2)


def test_floor_rejects_zero_repetitions():
    cell = encode_cells({"two": [["edit"], ["edit"]]})[0]
    with pytest.raises(ValueError, match="positive"):
        split_half_floor(cell, reps=0)


def test_pooled_forecast_excludes_held_out_transitions():
    cell = encode_cells({"unique": [["a"], ["b"], ["c"], ["d"]]})[0]
    pooled = cell.table()
    original = pooled.copy()
    scores = next_action_scores(cell, pooled, seed=0, alpha=0.5)
    assert scores["own"]["cross_entropy_bits"] == pytest.approx(3.0)
    assert scores["pooled"] == scores["own"]
    assert scores["n_test_transitions"]["value"] == 2
    np.testing.assert_array_equal(pooled, original)


def test_forecast_requires_a_pool_containing_the_cell():
    cell = encode_cells({"two": [["edit"], ["run_test"]]})[0]
    with pytest.raises(ValueError, match=r"include.*cell"):
        next_action_scores(cell, np.zeros_like(cell.table()))


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_precedence_intervals_match_scipy_wilson(n):
    confidence = float(2 * norm.cdf(1.96) - 1)
    for k in range(n + 1):
        runs = [["a", "b"]] * k + [["b", "a"]] * (n - k)
        measured = precedence_poset(runs)["pairs"]["a<b"]
        expected = binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
        assert measured["lo"] == pytest.approx(expected.low)
        assert measured["hi"] == pytest.approx(expected.high)
        assert 0 <= measured["lo"] <= measured["hi"] <= 1


def test_precedence_is_invariant_to_run_order():
    runs = [["a", "b", "c"]] * 10 + [["a", "c", "b"]]
    assert precedence_poset(runs) == precedence_poset(list(reversed(runs)))


@pytest.mark.parametrize("z", [0, -1, float("nan"), float("inf")])
def test_precedence_rejects_invalid_interval_width(z):
    with pytest.raises(ValueError, match="z must"):
        precedence_poset([["a", "b"]], z=z)


def test_poset_distance_rejects_no_shared_pairs():
    with pytest.raises(ValueError, match="shared observed action pairs"):
        poset_distance(precedence_poset([["a"]]), precedence_poset([["b"]]))


def test_conditional_distance_has_zero_and_positive_controls():
    left, right = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
    assert conditional_distance(left, left)[0] == 0
    distance, contributions = conditional_distance(left, right)
    assert distance == pytest.approx(1.0)
    assert contributions.sum() == pytest.approx(distance)
    assert unshared_mass(left, right) == 0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_conditional_distance_rejects_invalid_counts(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        conditional_distance(np.array([[value, 1.0]]), np.ones((1, 2)))


def test_conditional_distance_rejects_axis_mismatch():
    with pytest.raises(ValueError, match="same two-dimensional shape"):
        conditional_distance(np.ones((1, 2)), np.ones((2, 2)))


def test_unshared_mass_distinguishes_disjoint_states_from_missing_data():
    left, right = np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])
    assert unshared_mass(left, right) == 1
    with pytest.raises(ValueError, match="observations"):
        unshared_mass(left, np.zeros_like(left))


def test_distance_matrix_rejects_independently_encoded_axes():
    left = encode_cells({"a": [["edit"]]})[0]
    right = encode_cells({"b": [["run_test"]]})[0]
    with pytest.raises(ValueError, match="same ordered"):
        distance_matrix([left, right])


def test_distance_matrix_is_invariant_to_cell_order(cells):
    encoded = encode_cells(cells)
    expected = distance_matrix(encoded)
    np.testing.assert_allclose(distance_matrix(list(reversed(encoded))), expected[::-1, ::-1])


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_forecast_rejects_invalid_smoothing(alpha):
    cell = encode_cells({"two": [["edit"], ["run_test"]]})[0]
    with pytest.raises(ValueError, match="alpha must"):
        next_action_scores(cell, cell.table(), alpha=alpha)


def test_forecast_rejects_an_empty_half():
    cell = encode_cells({"empty": [[], []], "observed": [["edit"], ["edit"]]})[0]
    with pytest.raises(ValueError, match="observations"):
        next_action_scores(cell, cell.table())
